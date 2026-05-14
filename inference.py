import os
import argparse

BASE          = r'src/Reproduced Code/Image, Audio, Video/experiment_scripts'
CKPT          = r'src/Reproduced Code/Image, Audio, Video/logs'
BASE_IMPROVED = r'src/Improved Code/Image, Audio, Video/experiment_scripts'
CKPT_IMPROVED = r'src/Improved Code/Image, Audio, Video/logs'


def run(command):
    print(f"\n>>> {command}\n")
    os.system(command)


# ── Image ──────────────────────────────────────────────────────────────────────

def infer_image(model_type='sine', checkpoint=None):
    names = {
        'sine': 'siren_image',
        'relu': 'relu_image',
        'tanh': 'tanh_image',
        'rbf':  'rbf_image',
        'nerf': 'nerf_image',
    }
    folder = names.get(model_type, f'{model_type}_image')
    checkpoint = checkpoint or f'{CKPT}/{folder}/checkpoints/model_current.pth'
    run(
        f'python "{BASE}/train_img.py" '
        f'--model_type={model_type} '
        f'--experiment_name={folder}_inference '
        f'--logging_root=./results '
        f'--checkpoint_path="{checkpoint}" '
        f'--num_epochs=1'
    )


# ── Audio ──────────────────────────────────────────────────────────────────────

def infer_audio(model_type='sine', wav='./data/gt_bach.wav',
                experiment_name=None, checkpoint=None):
    experiment_name = experiment_name or f'audio_{model_type}_inference'
    checkpoint = checkpoint or f'{CKPT}/{experiment_name.replace("_inference","")}/checkpoints/model_current.pth'
    run(
        f'python "{BASE}/test_audio.py" '
        f'--model_type={model_type} '
        f'--gt_wav_path={wav} '
        f'--experiment_name={experiment_name} '
        f'--logging_root=./results '
        f'--checkpoint_path="{checkpoint}"'
    )


# ── Video ──────────────────────────────────────────────────────────────────────

def infer_video(model_type='sine', checkpoint=None):
    folder = 'video_siren' if model_type == 'sine' else f'video_{model_type}'
    checkpoint = checkpoint or f'{CKPT}/{folder}/checkpoints/model_current.pth'
    run(
        f'python "{BASE}/train_video.py" '
        f'--model_type={model_type} '
        f'--experiment_name={folder}_inference '
        f'--logging_root=./results '
        f'--checkpoint_path="{checkpoint}" '
        f'--num_epochs=1'
    )


# ── Improved Image Inference ───────────────────────────────────────────────────

def infer_img_caos(checkpoint=None):
    checkpoint = checkpoint or f'{CKPT_IMPROVED}/improved_image_caos/checkpoints/model_current.pth'
    run(
        f'python "{BASE_IMPROVED}/train_img_caos.py" '
        f'--experiment_name=improved_image_caos_inference '
        f'--logging_root=./results '
        f'--checkpoint_path="{checkpoint}" '
        f'--num_epochs=1'
    )


def infer_img_ffl(checkpoint=None):
    checkpoint = checkpoint or f'{CKPT_IMPROVED}/improved_image_ffl/checkpoints/model_current.pth'
    run(
        f'python "{BASE_IMPROVED}/train_img_ffl.py" '
        f'--experiment_name=improved_image_ffl_inference '
        f'--logging_root=./results '
        f'--checkpoint_path="{checkpoint}" '
        f'--num_epochs=1'
    )


def infer_img_grad_loss(checkpoint=None):
    checkpoint = checkpoint or f'{CKPT_IMPROVED}/improved_image_grad_loss/checkpoints/model_current.pth'
    run(
        f'python "{BASE_IMPROVED}/train_img_grad_loss.py" '
        f'--experiment_name=improved_image_grad_loss_inference '
        f'--logging_root=./results '
        f'--checkpoint_path="{checkpoint}" '
        f'--num_epochs=1'
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference for SIREN experiments')
    parser.add_argument(
        '--task', type=str, required=True,
        choices=[
            'image', 'audio', 'video',
            'img_caos', 'img_ffl', 'img_grad_loss',
        ],
        help=(
            'image            – reproduced image inference (use --model_type)\n'
            'audio            – reproduced audio inference (use --model_type, --wav, --name)\n'
            'video            – reproduced video inference (use --model_type)\n'
            'img_caos         – improved CAOS image inference\n'
            'img_ffl          – improved FFL image inference\n'
            'img_grad_loss    – improved gradient-loss image inference'
        )
    )
    parser.add_argument('--model_type', type=str, default='sine',
                        choices=['sine', 'relu', 'tanh', 'rbf', 'nerf'],
                        help='Model/activation type to use (reproduced experiments only)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (.pth). Defaults to the standard logs location.')
    parser.add_argument('--wav', type=str, default='./data/gt_bach.wav',
                        help='Path to ground-truth .wav file (audio task only)')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name override (audio task only)')

    args = parser.parse_args()

    if args.task == 'image':
        infer_image(args.model_type, args.checkpoint)
    elif args.task == 'audio':
        infer_audio(args.model_type, args.wav, args.name, args.checkpoint)
    elif args.task == 'video':
        infer_video(args.model_type, args.checkpoint)
    elif args.task == 'img_caos':
        infer_img_caos(args.checkpoint)
    elif args.task == 'img_ffl':
        infer_img_ffl(args.checkpoint)
    elif args.task == 'img_grad_loss':
        infer_img_grad_loss(args.checkpoint)