# SIREN — Reproduction of Results
### Assignment 2 | Department of AI & Data Science | FAST-NUCES

> **Paper:** Implicit Neural Representations with Periodic Activation Functions  
> **Authors:** Sitzmann et al., NeurIPS 2020  
> **Official Repo:** https://github.com/vsitzmann/siren

---

## Group ID09

| Name | Roll Number | Assigned Experiments |
|------|-------------|----------------------|
| Valeena Afzal | 25I-8023 | Image, Audio, Video |
| Maryam Zafar | 25I-8033 | Poisson, SDF |
| Laiba Noor | 25I-8035 | Helmholtz, Wave Equation |

**Instructor:** Dr. Zohair Ahmed | **Program:** BS DS/AI

---

## What is SIREN?

SIREN (Sinusoidal Representation Network) is a neural network that uses **sine** as its activation function instead of ReLU. This allows it to naturally represent signals and their derivatives, making it ideal for solving partial differential equations (PDEs) and fitting complex continuous signals.

```
φᵢ(xᵢ) = sin(ω₀ · (Wᵢxᵢ + bᵢ))     where ω₀ = 30
```

---

## Repository Structure

```
siren/                          ← Official repo (cloned)
│
├── experiment_scripts/
│   ├── train_helmholtz.py      ← Laiba's experiment 1
│   ├── train_wave_equation.py  ← Laiba's experiment 2
│   ├── train_img.py            ← Valeena's experiment
│   ├── train_audio.py          ← Valeena's experiment
│   ├── train_video.py          ← Valeena's experiment
│   ├── train_poisson_*.py      ← Maryam's experiments
│   └── train_sdf.py            ← Maryam's experiment
│
├── dataio.py                   ← Dataset classes
├── modules.py                  ← SIREN architecture
├── loss_functions.py           ← PDE loss functions
├── training.py                 ← Training loop
├── diff_operators.py           ← Gradient/Laplacian operators
├── utils.py                    ← TensorBoard summaries
└── torchmeta/                  ← Bundled (do NOT pip install)

notebooks/
├── DL_assignment_2_part_1.ipynb   ← Helmholtz experiment notebook
└── DL_assignment_2_part_2.ipynb   ← Wave equation experiment notebook

results/
├── helmholtz_loss_curve.png
├── helmholtz_wavefield.png
├── wave_loss_curve.png
└── wave_wavefield.png
```

---

## Setup (Google Colab — Recommended)

**Step 1 — Clone repo**
```python
!git clone https://github.com/vsitzmann/siren.git
%cd siren
```

**Step 2 — Install missing libraries**
```python
!pip install scikit-video cmapy configargparse
```

**Step 3 — Apply bug fixes**
```python
import subprocess

# Fix 1: Class name typo in utils.py
subprocess.run(['sed', '-i',
    's/NeuralProcessImplicit2DHypernetBVP/NeuralProcessImplicit2DHypernet/g',
    'utils.py'])

# Fix 2: Comment out summary_fn to prevent CUDA OOM (wave equation)
subprocess.run(['sed', '-i',
    's/summary_fn(model, model_input, gt, model_output, writer, total_steps)/#summary_fn(model, model_input, gt, model_output, writer, total_steps)/g',
    'training.py'])

# Fix 3: Reduce wave model size for 14GB GPU
subprocess.run(['sed', '-i',
    's/hidden_features=512/hidden_features=256/g',
    'experiment_scripts/train_wave_equation.py'])

print("All fixes applied!")
```

---

## Running Experiments

### Helmholtz Equation (Laiba)
```python
!python experiment_scripts/train_helmholtz.py \
    --experiment_name helmholtz_repro \
    --logging_root ./logs \
    --num_epochs 5000
```

### Wave Equation (Laiba)
```python
!python experiment_scripts/train_wave_equation.py \
    --experiment_name wave_repro \
    --logging_root ./logs \
    --num_epochs 5000 \
    --batch_size 3000
```

### Image Fitting (Valeena)
```python
!python experiment_scripts/train_img.py \
    --model_type sine \
    --experiment_name image_repro \
    --logging_root ./logs
```

### Audio Fitting (Valeena)
```python
!python experiment_scripts/train_audio.py \
    --model_type sine \
    --wav_path data/gt_bach.wav \
    --experiment_name audio_repro \
    --logging_root ./logs
```

### Poisson Equation (Maryam)
```python
!python experiment_scripts/train_poisson_grad_img.py \
    --experiment_name poisson_repro \
    --logging_root ./logs
```

---

## Libraries Used

| Library | Version | How Installed | Purpose |
|---------|---------|---------------|---------|
| PyTorch | Pre-installed | Colab default | Core framework |
| NumPy | 2.0.2 | Colab default | Array operations |
| Matplotlib | 3.10.0 | Colab default | Plotting |
| SciPy | 1.16.3 | Colab default | Differential operators |
| TensorBoard | 2.19.0 | Colab default | Training monitoring |
| scikit-video | 1.1.11 | `pip install scikit-video` | Video I/O |
| cmapy | 0.6.6 | `pip install cmapy` | Colormaps |
| opencv-python | 4.13.0.92 | Colab default | Image processing |
| scikit-image | 0.25.2 | Colab default | Image utilities |
| configargparse | 1.7.5 | `pip install configargparse` | Argument parsing |
| torchmeta | bundled | Local repo folder | Hypernetworks |

> **Note:** Do NOT `pip install torchmeta` — it is incompatible with modern PyTorch. Use the bundled `./torchmeta/` folder instead.

---

## Results

### Helmholtz Equation

| Metric | Paper | Ours |
|--------|-------|------|
| Training Steps | 50,000 | 5,000 |
| Initial Loss | — | 5,581,349 |
| Final Loss | Converged | 175,564 |
| Loss Reduction | ~96%+ | 96.9% |


### Wave Equation

| Metric | Paper | Ours |
|--------|-------|------|
| Training Steps | 100,000 | 5,000 |
| Initial Loss | — | 85,955,296 |
| Final Loss | Converged | 16,236 |
| Loss Reduction | — | 99.98% |


---

## Known Issues & Fixes

### 1. `AttributeError: NeuralProcessImplicit2DHypernetBVP`
**Cause:** Typo in `utils.py` — class does not exist  
**Fix:** Replace with `NeuralProcessImplicit2DHypernet` (see setup Step 3)

### 2. `CUDA Out of Memory` — Wave Equation
**Cause:** `summary_fn()` computes Jacobians for visualization, consuming 13+ GB  
**Fix:** Comment out `summary_fn()` in `training.py` (see setup Step 3)  
**Reference:** GitHub Issue [#20](https://github.com/vsitzmann/siren/issues/20)

### 3. `torchmeta` pip install fails
**Cause:** All versions require `torch < 1.10`, incompatible with modern PyTorch  
**Fix:** Use the bundled `./torchmeta/` folder — do not pip install

### 4. `FileExistsError` when re-running experiments
**Cause:** Log folder already exists from a previous run  
**Fix:** `!rm -rf ./logs/<experiment_name>` before re-running

### 5. Large Helmholtz loss values (millions)
**Cause:** Loss is summed over all sampled points, not averaged  
**Note:** This is expected and confirmed by paper authors in GitHub Issue [#7](https://github.com/vsitzmann/siren/issues/7)

---

## Hardware

- **Platform:** Google Colab (free tier)
- **GPU:** NVIDIA T4 (14.56 GB VRAM)
- **Training time:** ~63 min (Helmholtz, 5000 steps) | ~67 min (Wave, 5000 steps)

---

## Citation

```bibtex
@inproceedings{sitzmann2019siren,
    author    = {Sitzmann, Vincent and Martel, Julien N.P. and
                 Bergman, Alexander W. and Lindell, David B. and Wetzstein, Gordon},
    title     = {Implicit Neural Representations with Periodic Activation Functions},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2020}
}
```

---

## References

1. Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions," NeurIPS 2020
2. Official SIREN GitHub: https://github.com/vsitzmann/siren
3. GitHub Issue #7 (Helmholtz loss): https://github.com/vsitzmann/siren/issues/7
4. GitHub Issue #20 (Wave OOM): https://github.com/vsitzmann/siren/issues/20


# Assignment 3: Experimentation & Expansion — SIREN

> **Base Paper:** *Implicit Neural Representations with Periodic Activation Functions*
> Sitzmann et al., NeurIPS 2020 · [arXiv:2006.09661](https://arxiv.org/abs/2006.09661) · [Official Repo](https://github.com/vsitzmann/siren)

FAST-NUCES | Department of AI & Data Science · Group ID09 · April 2026

---

## Group Members

| Name | Roll No. | Assigned Part |
|---|---|---|
| Valeena Afzal | 25I-8023 | Improvement I & II — CAOS-SIREN + Fourier Frequency Loss + Gradient Supervision Loss |
| Maryam Zafar | 25I-8033 | Improvement III — H-SIREN Activation + SDF on Stanford Bunny |
| Laiba Noor | 25I-8035 | Improvement IV + Bonus — Helmholtz k-Ablation, Wave IC Study, Brain MRI SR |

**Instructor:** Dr. Zohair Ahmed · **Program:** BS DS/AI

---

## Overview

This assignment extends the SIREN reproduction from A2 with four original improvements and one bonus experiment, testing whether targeted modifications can outperform the fixed ω₀=30 baseline.

All four modifications failed to beat the baseline — establishing the key insight that SIREN's strength lies in its principled initialisation, not its training dynamics.

---

## Experiments

### Improvement I — CAOS-SIREN (Valeena)

Cosine-annealed ω₀ schedule instead of fixed ω₀=30.

```
ω(t) = ω_min + ½(ω_max − ω_min)(1 − cos(πt/T))
```

| Method | ω Range | PSNR | vs Baseline |
|---|---|---|---|
| SIREN baseline | Fixed 30 | 40.37 dB | — |
| CAOS Run 1 | 10 → 60 | 29.77 dB | −10.60 dB |
| CAOS Run 2 | 25 → 45 | 19.59 dB | −20.78 dB |

**Finding:** PSNR rises then collapses — increasing ω mid-training invalidates the uniform initialisation assumption.

---

### Improvement II — Fourier Frequency Loss (Valeena)

Augments L2 loss with a spectral L1 penalty: `L = L2 + λ · ‖FFT(ŷ) − FFT(y)‖₁`

| λ | PSNR | Final Loss |
|---|---|---|
| Baseline | 40.37 dB | — |
| 0.01 | 29.37 dB | 0.0387 |
| 0.05 | 26.77 dB | 0.1745 |
| 0.10 | 25.77 dB | 0.3541 |

**Finding:** SIREN already performs implicit Fourier decomposition via sine activations — explicit FFT supervision fights the network's natural frequency learning.

---

### Improvement III — Gradient Supervision Loss (Valeena)

Adds auxiliary gradient-matching: `L = L2 + λ · ‖∇ŷ − ∇y‖₁` with 1000-step linear warmup.

| λ | PSNR | SSIM |
|---|---|---|
| Baseline | 40.37 dB | — |
| 0.01 | 26.14 dB | 0.59 |
| 0.05 | 21.48 dB | 0.38 |
| 0.10 | 21.47 dB | 0.31 |

**Finding:** Gradient matching competes with pixel accuracy. λ=0.01 is closest to baseline but still −14 dB below.

---

### Improvement IV — H-SIREN (Maryam)

Replaces the first-layer activation with `sin(sinh(2x))` for a broader initial frequency range.

**Poisson Reconstruction:**

| Model | Image | PSNR | MSE |
|---|---|---|---|
| SIREN | Cameraman | 33.46 dB | 0.000451 |
| H-SIREN | Cameraman | 33.41 dB | 0.000456 |
| SIREN | Coffee | 31.94 dB | 0.000640 |
| H-SIREN | Coffee | 26.83 dB | 0.002076 |

**SDF on Stanford Bunny:**

| Model | Vertices | Faces | Final Loss | Result |
|---|---|---|---|---|
| SIREN | 50,083 | 100,156 | 4.16e+00 | Valid SDF |
| H-SIREN | 401,006 | 876,634 | 1.51e−01 | Collapsed |

**Finding:** H-SIREN works on simple images (−0.05 dB) but diverges on complex images and catastrophically collapses on SDF — `sin(sinh(2x))` breaks the Eikonal-loss-required initialisation statistics.

---

### Improvement IV-B — Helmholtz k-Wavenumber Ablation (Laiba)

The original paper uses a single fixed k. This experiment ablates k ∈ {1, 5, 20, 50}.

| k | Initial Loss | Final Loss (1K steps) | Reduction | Steps to 95% |
|---|---|---|---|---|
| 1 | 8,543,346 | 261 | 100% | 15 |
| 5 | 6,551,798 | 166 | 100% | 14 |
| 20 | 17,054,916 | 586 | 100% | 19 |
| 50 | 107,369,632 | 24,346 | 100% | 22 |

**Finding:** SIREN achieves 100% loss reduction at every wavenumber k=1 to k=50. The original paper's single-k evaluation under-represents SIREN's frequency range capability.

---

### Improvement IV-C — Wave Equation Initial Condition Study (Laiba)

Original paper fixes IC-1: `u(x,0) = sin(πx)`. Two new ICs introduced.

| Initial Condition | Type | Initial Loss | Final Loss (500 steps) | Reduction |
|---|---|---|---|---|
| IC-1: sin(πx) | Baseline | 71,598 | 423 | 99.4% |
| IC-2: exp(−‖x‖²/0.05) | Gaussian pulse | 56,604 | 116 | 99.8% |
| IC-3: sin(2πx)+0.5sin(4πx) | Double sinusoid | 70,266 | 492 | 99.3% |

**Finding:** Gaussian pulse converges 3.6x faster to a 3.6x lower final loss. All ICs produce physically plausible spacetime solutions.

---

### Bonus — Brain MRI Super-Resolution (Laiba)

SIREN applied to T1-weighted sagittal slices from the [IXI dataset](https://brain-development.org/ixi-dataset/), downsampled 2x and 4x.

| Method | Scale | PSNR (dB) | SSIM |
|---|---|---|---|
| Bicubic | 2x | 26.99 ± 0.54 | 0.883 |
| ReLU MLP | 2x | 19.31 ± 0.47 | 0.372 |
| SIREN (A2) | 2x | 26.33 ± 0.55 | 0.865 |
| SIREN+FFL+CAOS | 2x | 14.20 ± 0.90 | 0.101 |
| Bicubic | 4x | 22.29 ± 0.56 | 0.625 |
| ReLU MLP | 4x | 19.05 ± 0.55 | 0.327 |
| SIREN (A2) | 4x | 21.40 ± 0.60 | 0.549 |
| SIREN+FFL+CAOS | 4x | 13.33 ± 1.00 | 0.056 |

**Finding:** SIREN reaches within 0.66 dB of bicubic at 2x without any domain-specific modification. The combined SIREN+FFL+CAOS fails — FFT penalty conflicts with smooth MRI intensity distributions.

---

## Central Insight

All four modifications degraded SIREN performance. The root cause is the same in every case: each modification disrupts the principled uniform weight initialisation that SIREN's performance depends on. SIREN's strength comes from its initialisation and pure L2 objective, not from training dynamics or loss design.

---

## Setup

```bash
git clone https://github.com/vsitzmann/siren
cd siren

pip install torch==2.6.0 torchvision==0.21.0 numpy==2.0.2 \
    scikit-image==0.25.2 trimesh==3.22.0 nibabel tensorboard==2.19.0
```

### Hardware

| Member | Platform | GPU | VRAM |
|---|---|---|---|
| Laiba | Google Colab | T4 | 14.56 GB |
| Valeena | Local (Anaconda) | RTX 3050 | 4 GB |
| Maryam | Google Colab | T4 | 14.56 GB |

---

## Hyperparameters

| Experiment | Architecture | ω₀ | LR | Steps |
|---|---|---|---|---|
| Helmholtz k-ablation | 4×256 | 30 | 1e-4 | 1K per k |
| Wave IC variation | 4×256 | 30 | 2e-5 | 500 per IC |
| MRI SR (bonus) | 5×256 | 30 | 1e-4 | 5K per slice |
| Image / CAOS / FFL | 5×256 | 30 | 1e-4 | 10K |
| Poisson / SDF (H-SIREN) | 5×256 | 30 | 1e-4 / 2e-5 | 10K / 5K |

---

## Datasets

| Dataset | Used For | Source |
|---|---|---|
| BSD500 cameraman (512×512) | Image fitting, CAOS, FFL, Grad Loss | [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) |
| Kodak24 | ω₀ ablation | [Kaggle](https://kaggle.com/datasets/sherylmehta/kodak-dataset) |
| BSD68 (10 images) | FFL evaluation | [GitHub](https://github.com/clausmichele/CBSD68-dataset) |
| BSD500 "coffee" image | Poisson 2nd test | Subset of A2 dataset |
| Stanford Bunny .ply (500K pts) | SDF (H-SIREN) | [Stanford 3D](https://graphics.stanford.edu/data/3Dscanrep/) |
| IXI Brain MRI T1 (5 slices) | Bonus MRI SR | [IXI Dataset](https://brain-development.org/ixi-dataset/) (CC license) |

---

## References

1. Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions," NeurIPS 2020
2. Gao & Jaiman, "H-SIREN," [arXiv:2410.04716](https://arxiv.org/abs/2410.04716), 2024
3. Zhao et al., "Focal Frequency Loss for Image Restoration," [arXiv:2012.12821](https://arxiv.org/abs/2012.12821), 2021
4. [Official SIREN GitHub](https://github.com/vsitzmann/siren)
5. [GitHub Issue #7 — Helmholtz loss](https://github.com/vsitzmann/siren/issues/7)



---------
# Run all experiments at once
python train.py --task all

# Individual modalities
python train.py --task image_all
python train.py --task audio_all
python train.py --task video_all

# Single experiment with a specific model type
python train.py --task image --model_type relu
python train.py --task video --model_type sine

# Audio with custom wav and name
python train.py --task audio --model_type sine --wav ./data/gt_bach.wav --name audio_bach_siren


----
# Image inference (defaults to logs/siren_image/checkpoints/model_current.pth)
python inference.py --task image --model_type sine

# Audio inference with a custom checkpoint
python inference.py --task audio --model_type sine --wav ./data/gt_bach.wav --checkpoint path/to/model.pth

# Video inference
python inference.py --task video --model_type relu

----
# Run individually
python train.py --task img_caos
python train.py --task img_ffl
python train.py --task img_grad_loss

# Run all three improved experiments
python train.py --task improved_all

# Run everything (reproduced + improved)
python train.py --task all

