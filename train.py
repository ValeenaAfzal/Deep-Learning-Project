import os
import argparse

BASE          = r'src/Reproduced Code/Image, Audio, Video/experiment_scripts'
BASE_IMPROVED = r'src/Improved Code/Image, Audio, Video/experiment_scripts'
LOGS          = './logs'


def run(command):
    print(f"\n>>> {command}\n")
    os.system(command)


# ── Image ──────────────────────────────────────────────────────────────────────

def train_image(model_type='sine'):
    names = {
        'sine':    'siren_image',
        'relu':    'relu_image',
        'tanh':    'tanh_image',
        'rbf':     'rbf_image',
        'nerf':    'nerf_image',
    }
    experiment_name = names.get(model_type, f'{model_type}_image')
    run(
        f'python "{BASE}/train_img.py" '
        f'--model_type={model_type} '
        f'--experiment_name={experiment_name} '
        f'--logging_root={LOGS} '
        f'--steps_til_summary=1000'
    )


def train_all_images():
    for model_type in ['sine', 'relu', 'tanh', 'rbf', 'nerf']:
        train_image(model_type)


# ── Audio ──────────────────────────────────────────────────────────────────────

def train_audio(model_type, wav, experiment_name):
    run(
        f'python "{BASE}/train_audio.py" '
        f'--model_type={model_type} '
        f'--wav_path={wav} '
        f'--experiment_name={experiment_name} '
        f'--logging_root={LOGS}'
    )


def train_all_audio():
    experiments = [
        ('sine', './data/gt_bach.wav',      'audio_bach_siren'),
        ('relu', './data/gt_bach.wav',      'audio_bach_relu'),
        ('sine', './data/gt_counting.wav',  'audio_counting_siren'),
        ('relu', './data/gt_counting.wav',  'audio_counting_relu'),
    ]
    for model_type, wav, name in experiments:
        train_audio(model_type, wav, name)


# ── Video ──────────────────────────────────────────────────────────────────────

def train_video(model_type='sine'):
    configs = {
        'sine': ('video_siren', 5000),
        'relu': ('video_relu', 10000),
    }
    experiment_name, num_epochs = configs.get(model_type, (f'video_{model_type}', 10000))
    run(
        f'python "{BASE}/train_video.py" '
        f'--model_type={model_type} '
        f'--experiment_name={experiment_name} '
        f'--logging_root={LOGS} '
        f'--num_epochs={num_epochs}'
    )


def train_all_video():
    for model_type in ['sine', 'relu']:
        train_video(model_type)


# ── Improved Image Experiments ─────────────────────────────────────────────────

def train_img_caos():
    run(
        f'python "{BASE_IMPROVED}/train_img_caos.py" '
        f'--experiment_name=improved_image_caos '
        f'--logging_root={LOGS} '
        f'--steps_til_summary=1000'
    )


def train_img_ffl():
    run(
        f'python "{BASE_IMPROVED}/train_img_ffl.py" '
        f'--experiment_name=improved_image_ffl '
        f'--logging_root={LOGS} '
        f'--steps_til_summary=1000'
    )


def train_img_grad_loss():
    run(
        f'python "{BASE_IMPROVED}/train_img_grad_loss.py" '
        f'--experiment_name=improved_image_grad_loss '
        f'--logging_root={LOGS} '
        f'--steps_til_summary=1000'
    )


def train_all_improved():
    train_img_caos()
    train_img_ffl()
    train_img_grad_loss()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SIREN experiments')
    parser.add_argument(
        '--task', type=str, required=True,
        choices=[
            'image', 'image_all',
            'audio', 'audio_all',
            'video', 'video_all',
            'img_caos', 'img_ffl', 'img_grad_loss', 'improved_all',
            'all',
        ],
        help=(
            'image            – train a single image model (use --model_type)\n'
            'image_all        – train all image variants\n'
            'audio            – train a single audio experiment (use --model_type, --wav, --name)\n'
            'audio_all        – train all audio experiments\n'
            'video            – train a single video model (use --model_type)\n'
            'video_all        – train all video variants\n'
            'img_caos         – train improved CAOS image model\n'
            'img_ffl          – train improved FFL image model\n'
            'img_grad_loss    – train improved gradient-loss image model\n'
            'improved_all     – run all three improved image experiments\n'
            'all              – run every experiment sequentially (reproduced + improved)'
        )
    )
    parser.add_argument('--model_type', type=str, default='sine',
                        choices=['sine', 'relu', 'tanh', 'rbf', 'nerf'],
                        help='Activation / encoding type (used with image / audio / video tasks)')
    parser.add_argument('--wav', type=str, default='./data/gt_bach.wav',
                        help='Path to .wav file (used with --task audio)')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name override (used with --task audio)')

    args = parser.parse_args()

    if args.task == 'image':
        train_image(args.model_type)
    elif args.task == 'image_all':
        train_all_images()
    elif args.task == 'audio':
        name = args.name or f'audio_{args.model_type}'
        train_audio(args.model_type, args.wav, name)
    elif args.task == 'audio_all':
        train_all_audio()
    elif args.task == 'video':
        train_video(args.model_type)
    elif args.task == 'video_all':
        train_all_video()
    elif args.task == 'img_caos':
        train_img_caos()
    elif args.task == 'img_ffl':
        train_img_ffl()
    elif args.task == 'img_grad_loss':
        train_img_grad_loss()
    elif args.task == 'improved_all':
        train_all_improved()
    elif args.task == 'all':
        train_all_images()
        train_all_audio()
        train_all_video()
        train_all_improved()