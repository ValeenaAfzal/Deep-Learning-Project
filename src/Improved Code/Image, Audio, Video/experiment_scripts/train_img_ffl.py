import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, loss_functions, modules
from torch.utils.data import DataLoader
from functools import partial
import torch
import numpy as np
import csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def ffl_loss(pred, gt, lambda_ffl=0.05, step=0, warmup_steps=2000):
    # L2 pixel loss
    l2 = ((pred - gt)**2).mean()
    # Lambda warmup
    lam = lambda_ffl * min(1.0, step / warmup_steps)
    if lam == 0:
        return l2
    # Reshape to image for FFT
    H, W = 512, 512
    pred_img = pred.reshape(H, W)
    gt_img = gt.reshape(H, W)
    # 2D FFT magnitude difference
    pred_fft = torch.fft.fft2(pred_img)
    gt_fft = torch.fft.fft2(gt_img)
    fft_l1 = torch.abs(torch.abs(pred_fft) - torch.abs(gt_fft)).mean()
    return l2 + lam * fft_l1

# Dataset
img_dataset = dataio.Camera()
coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=512, compute_diff='all')
image_resolution = (512, 512)
dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1,
                        pin_memory=True, num_workers=0)

lambda_values = [0.01, 0.05, 0.1]
results = {}

for lambda_ffl in lambda_values:
    print(f"\n{'='*50}")
    print(f"Training SIREN + FFL | lambda = {lambda_ffl}")
    print(f"{'='*50}")

    model = modules.SingleBVPNet(type='sine', mode='mlp', sidelength=image_resolution)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    exp_name = f'ffl_lambda_{str(lambda_ffl).replace(".", "_")}'
    os.makedirs(f'./logs/{exp_name}/summaries', exist_ok=True)
    os.makedirs(f'./logs/{exp_name}/checkpoints', exist_ok=True)

    tb_writer = SummaryWriter(f'./logs/{exp_name}/summaries')
    log_file = open(f'./logs/{exp_name}_log.csv', 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(['step', 'loss', 'psnr'])

    total_steps = 10000
    step = 0
    final_psnr = 0

    pbar = tqdm(total=total_steps, desc=f'lambda={lambda_ffl}')

    for epoch in range(total_steps):
        for model_input, gt in dataloader:
            model_input = {key: value.to(device) for key, value in model_input.items()}
            gt = {key: value.to(device) for key, value in gt.items()}

            model_output = model(model_input)
            pred = model_output['model_out']
            target = gt['img']

            train_loss = ffl_loss(pred, target, lambda_ffl=lambda_ffl, step=step)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                mse = ((pred.clamp(-1, 1) - target)**2).mean().item()
                psnr = 10 * np.log10(4.0 / mse) if mse > 0 else 100.0
                final_psnr = psnr

            if step % 500 == 0:
                csv_writer.writerow([step, f"{train_loss.item():.6f}", f"{psnr:.2f}"])
                tb_writer.add_scalar('train_loss', train_loss.item(), step)
                tb_writer.add_scalar('train_psnr', psnr, step)
                log_file.flush()
                print(f"Step {step} | Loss: {train_loss.item():.6f} | PSNR: {psnr:.2f} dB")

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

    results[lambda_ffl] = final_psnr
    print(f"lambda={lambda_ffl} | Final PSNR: {final_psnr:.2f} dB")

print("\n===== FFL RESULTS SUMMARY =====")
print(f"SIREN A2 Baseline: 40.37 dB")
for lam, psnr in results.items():
    diff = psnr - 40.37
    print(f"SIREN + FFL (lambda={lam}): {psnr:.2f} dB ({'+' if diff >= 0 else ''}{diff:.2f} dB vs baseline)")