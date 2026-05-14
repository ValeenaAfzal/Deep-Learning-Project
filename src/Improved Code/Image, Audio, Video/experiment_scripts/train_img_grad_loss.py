import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, loss_functions, modules
import diff_operators
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch
import numpy as np
import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import skimage.metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Gradient Loss ─────────────────────────────────────────────────────────────
def gradient_loss(model_output, gt, lambda_grad=0.1, step=0, warmup_steps=1000):
    """
    L_total = L2_pixel + lambda * ||grad(pred) - grad(gt)||
    
    Motivation: Standard L2 treats all pixels equally and may underrepresent
    edges and fine details. Adding explicit gradient supervision encourages
    the network to accurately reconstruct spatial derivatives — something
    SIREN is uniquely capable of due to its periodic activations.
    """
    pred   = model_output['model_out']
    target = gt['img']

    # L2 pixel loss
    l2_loss = ((pred - target)**2).mean()

    # Lambda warmup — gradually increase gradient weight
    lam = lambda_grad * min(1.0, step / warmup_steps)
    if lam == 0:
        return l2_loss, l2_loss.item(), 0.0

    # Predicted gradients via autograd
    pred_grad = diff_operators.gradient(model_output['model_out'],
                                        model_output['model_in'])

    # Ground truth gradients from dataset
    gt_grad = gt['gradients']

    # Gradient L1 loss — L1 is more robust than L2 for gradients
    grad_loss = torch.abs(pred_grad - gt_grad).mean()

    total = l2_loss + lam * grad_loss
    return total, l2_loss.item(), grad_loss.item()


# ── Dataset ───────────────────────────────────────────────────────────────────
img_dataset      = dataio.Camera()
coord_dataset    = dataio.Implicit2DWrapper(img_dataset, sidelength=512,
                                            compute_diff='all')
image_resolution = (512, 512)
dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1,
                        pin_memory=True, num_workers=0)

# ── Helper: full image summary ────────────────────────────────────────────────
def write_summary(model_input, gt, model_output, tb_writer, step, prefix='train_'):
    torch.cuda.empty_cache()
    with torch.no_grad():
        gt_img   = dataio.lin2img(gt['img'],                 image_resolution).cpu()
        pred_img = dataio.lin2img(model_output['model_out'], image_resolution).cpu()

        # GT vs Pred
        output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
        tb_writer.add_image(prefix + 'gt_vs_pred',
                            make_grid(output_vs_gt, scale_each=False, normalize=True),
                            global_step=step)

        pred_np = dataio.rescale_img((pred_img + 1) / 2, mode='clamp') \
                        .permute(0, 2, 3, 1).squeeze(0).numpy()
        gt_np   = dataio.rescale_img((gt_img + 1) / 2, mode='clamp') \
                        .permute(0, 2, 3, 1).squeeze(0).numpy()

        tb_writer.add_image(prefix + 'pred_img',
                            torch.from_numpy(pred_np).permute(2, 0, 1), global_step=step)
        tb_writer.add_image(prefix + 'gt_img',
                            torch.from_numpy(gt_np).permute(2, 0, 1),   global_step=step)

    # Gradient images
    try:
        pred_grad = dataio.grads2img(
            dataio.lin2img(diff_operators.gradient(
                model_output['model_out'], model_output['model_in']
            ))
        ).permute(1, 2, 0).squeeze().detach().cpu().numpy()
        tb_writer.add_image(prefix + 'pred_grad',
                            torch.from_numpy(pred_grad).permute(2, 0, 1),
                            global_step=step)

        gt_grad = dataio.grads2img(
            dataio.lin2img(gt['gradients'])
        ).permute(1, 2, 0).squeeze().detach().cpu().numpy()
        tb_writer.add_image(prefix + 'gt_grad',
                            torch.from_numpy(gt_grad).permute(2, 0, 1),
                            global_step=step)
    except Exception as e:
        print(f"  [grad image skipped: {e}]")

    # PSNR + SSIM
    try:
        p    = pred_np.squeeze()
        trgt = gt_np.squeeze()
        psnr_val = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)
        ssim_val = skimage.metrics.structural_similarity(
            p, trgt,
            channel_axis=-1 if p.ndim == 3 else None,
            data_range=1)
        tb_writer.add_scalar(prefix + 'img_psnr', psnr_val, global_step=step)
        tb_writer.add_scalar(prefix + 'img_ssim', ssim_val, global_step=step)
        return psnr_val, ssim_val
    except Exception as e:
        print(f"  [psnr/ssim skipped: {e}]")
        return None, None


# ── Train one model per lambda ────────────────────────────────────────────────
lambda_values = [0.01, 0.05, 0.1]
results       = {}

for lambda_grad in lambda_values:
    print(f"\n{'='*55}")
    print(f"  Training SIREN + Gradient Loss  |  lambda = {lambda_grad}")
    print(f"{'='*55}")

    model     = modules.SingleBVPNet(type='sine', mode='mlp',
                                     sidelength=image_resolution)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    exp_name = f'gradloss_lambda_{str(lambda_grad).replace(".", "_")}'
    os.makedirs(f'./logs/{exp_name}/summaries',   exist_ok=True)
    os.makedirs(f'./logs/{exp_name}/checkpoints', exist_ok=True)

    tb_writer  = SummaryWriter(f'./logs/{exp_name}/summaries')
    log_file   = open(f'./logs/{exp_name}_log.csv', 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(['step', 'total_loss', 'l2_loss', 'grad_loss', 'psnr', 'ssim'])

    total_steps = 10000
    step        = 0
    final_psnr  = 0

    pbar = tqdm(total=total_steps, desc=f'lambda={lambda_grad}')

    for epoch in range(total_steps):
        for model_input, gt in dataloader:
            model_input = {k: v.to(device) for k, v in model_input.items()}
            gt          = {k: v.to(device) for k, v in gt.items()}

            model_output = model(model_input)

            train_loss, l2_val, grad_val = gradient_loss(
                model_output, gt,
                lambda_grad=lambda_grad,
                step=step,
                warmup_steps=1000
            )

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = model_output['model_out'].clamp(-1, 1)
                mse  = ((pred - gt['img'])**2).mean().item()
                psnr = 10 * np.log10(4.0 / mse) if mse > 0 else 100.0
                final_psnr = psnr

            if step % 500 == 0:
                tb_writer.add_scalar('train_total_loss', train_loss.item(), step)
                tb_writer.add_scalar('train_l2_loss',    l2_val,            step)
                tb_writer.add_scalar('train_grad_loss',  grad_val,          step)
                tb_writer.add_scalar('train_psnr',       psnr,              step)

                psnr_val, ssim_val = write_summary(
                    model_input, gt, model_output,
                    tb_writer, step, prefix='train_')

                csv_writer.writerow([
                    step,
                    f"{train_loss.item():.6f}",
                    f"{l2_val:.6f}",
                    f"{grad_val:.6f}",
                    f"{psnr:.2f}",
                    f"{ssim_val:.4f}" if ssim_val else ""
                ])
                log_file.flush()
                print(f"  Step {step:5d} | Total: {train_loss.item():.6f} | "
                      f"L2: {l2_val:.6f} | Grad: {grad_val:.6f} | "
                      f"PSNR: {psnr:.2f} dB")

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
               f'./logs/{exp_name}/checkpoints/model_final.pth')

    results[lambda_grad] = final_psnr
    print(f"  lambda={lambda_grad} | Final PSNR: {final_psnr:.2f} dB")


# ── Final summary ─────────────────────────────────────────────────────────────
print("\n===== GRADIENT LOSS RESULTS SUMMARY =====")
print(f"SIREN A2 Baseline: 40.37 dB")
for lam, psnr in results.items():
    diff = psnr - 40.37
    sign = '+' if diff >= 0 else ''
    print(f"SIREN + GradLoss (lambda={lam}): {psnr:.2f} dB  "
          f"({sign}{diff:.2f} dB vs baseline)")