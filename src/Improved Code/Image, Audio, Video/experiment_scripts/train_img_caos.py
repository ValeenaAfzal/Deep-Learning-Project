import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, utils, loss_functions, modules
import diff_operators
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from functools import partial
import torch
import numpy as np
import math
import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import skimage.metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── CAOS Schedule ─────────────────────────────────────────────────────────────
def caos_omega(step, total_steps, omega_min=25, omega_max=45):
    """Cosine Annealed Omega Schedule"""
    return omega_min + (omega_max - omega_min) * (1 - math.cos(math.pi * step / total_steps)) / 2

# ── Dataset ───────────────────────────────────────────────────────────────────
img_dataset   = dataio.Camera()
coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=512, compute_diff='all')
image_resolution = (512, 512)
dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1,
                        pin_memory=True, num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────────
model = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=image_resolution)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn   = partial(loss_functions.image_mse, None)

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs('./logs/caos_siren/summaries',   exist_ok=True)
os.makedirs('./logs/caos_siren/checkpoints', exist_ok=True)
tb_writer  = SummaryWriter('./logs/caos_siren/summaries')
log_file   = open('./logs/caos_siren_log.csv', 'w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(['step', 'omega_0', 'loss', 'psnr', 'ssim'])

# ── Helper: write full image summary (like original experiments) ───────────────
def write_summary(model_input, gt, model_output, tb_writer, step, prefix='train_'):
    torch.cuda.empty_cache()
    with torch.no_grad():
        gt_img   = dataio.lin2img(gt['img'],             image_resolution).cpu()
        pred_img = dataio.lin2img(model_output['model_out'], image_resolution).cpu()

        # GT vs Pred side by side
        output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
        tb_writer.add_image(prefix + 'gt_vs_pred',
                            make_grid(output_vs_gt, scale_each=False, normalize=True),
                            global_step=step)

        # Individual images (rescaled to [0,1])
        pred_np = dataio.rescale_img((pred_img + 1) / 2, mode='clamp') \
                        .permute(0, 2, 3, 1).squeeze(0).numpy()
        gt_np   = dataio.rescale_img((gt_img   + 1) / 2, mode='clamp') \
                        .permute(0, 2, 3, 1).squeeze(0).numpy()

        tb_writer.add_image(prefix + 'pred_img',
                            torch.from_numpy(pred_np).permute(2, 0, 1), global_step=step)
        tb_writer.add_image(prefix + 'gt_img',
                            torch.from_numpy(gt_np).permute(2, 0, 1),   global_step=step)

    # Gradient visualization (requires grad — do outside no_grad)
    try:
        img_gradient = diff_operators.gradient(
            model_output['model_out'], model_output['model_in'])
        pred_grad = dataio.grads2img(dataio.lin2img(img_gradient)) \
                          .permute(1, 2, 0).squeeze().detach().cpu().numpy()
        tb_writer.add_image(prefix + 'pred_grad',
                            torch.from_numpy(pred_grad).permute(2, 0, 1), global_step=step)

        gt_grad = dataio.grads2img(dataio.lin2img(gt['gradients'])) \
                        .permute(1, 2, 0).squeeze().detach().cpu().numpy()
        tb_writer.add_image(prefix + 'gt_grad',
                            torch.from_numpy(gt_grad).permute(2, 0, 1), global_step=step)
    except Exception as e:
        print(f"  [grad summary skipped: {e}]")

    # PSNR + SSIM
    try:
        p    = pred_np.squeeze()
        trgt = gt_np.squeeze()
        psnr_val = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)
        ssim_val = skimage.metrics.structural_similarity(
            p, trgt, channel_axis=-1 if p.ndim == 3 else None, data_range=1)
        tb_writer.add_scalar(prefix + 'img_psnr', psnr_val, global_step=step)
        tb_writer.add_scalar(prefix + 'img_ssim', ssim_val, global_step=step)
        return psnr_val, ssim_val
    except Exception as e:
        print(f"  [psnr/ssim skipped: {e}]")
        return None, None

# ── Training loop ─────────────────────────────────────────────────────────────
total_steps = 10000
step        = 0
final_psnr  = 0

print("Starting CAOS-SIREN training...")
pbar = tqdm(total=total_steps)

for epoch in range(total_steps):
    for model_input, gt in dataloader:

        # Update omega every 500 steps
        if step % 500 == 0:
            new_omega = caos_omega(step, total_steps)
            model.update_omega(new_omega)
            print(f"\nStep {step}: omega_0 updated to {new_omega:.2f}")

        model_input = {k: v.to(device) for k, v in model_input.items()}
        gt          = {k: v.to(device) for k, v in gt.items()}

        model_output = model(model_input)
        losses       = loss_fn(model_output, gt)
        train_loss   = sum(losses.values())

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Quick PSNR estimate
        with torch.no_grad():
            pred = model_output['model_out'].clamp(-1, 1)
            mse  = ((pred - gt['img'])**2).mean().item()
            psnr = 10 * np.log10(4.0 / mse) if mse > 0 else 100.0
            final_psnr = psnr

        # Full summary every 500 steps
        if step % 500 == 0:
            current_omega = caos_omega(step, total_steps)
            tb_writer.add_scalar('train_loss',  train_loss.item(), step)
            tb_writer.add_scalar('train_psnr',  psnr,             step)
            tb_writer.add_scalar('omega_0',     current_omega,     step)

            # Full image summary
            psnr_val, ssim_val = write_summary(model_input, gt, model_output,
                                               tb_writer, step, prefix='train_')

            csv_writer.writerow([step,
                                 f"{current_omega:.2f}",
                                 f"{train_loss.item():.6f}",
                                 f"{psnr:.2f}",
                                 f"{ssim_val:.4f}" if ssim_val else ""])
            log_file.flush()
            print(f"Step {step:5d} | Loss: {train_loss.item():.6f} | "
                  f"PSNR: {psnr:.2f} dB | omega: {current_omega:.2f}")

        pbar.update(1)
        step += 1
        if step >= total_steps:
            break
    if step >= total_steps:
        break

pbar.close()
log_file.close()
tb_writer.close()
torch.save({'model': model.state_dict()},
           './logs/caos_siren/checkpoints/model_final.pth')
print(f"\nFinal PSNR: {final_psnr:.2f} dB")
print("CAOS-SIREN training complete!")
