import os
import argparse
import random

import shutil
from pathlib import Path
import clip.clip as openai_clip
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_encoder_architecture_usage
from datasets import get_shadow_dataset
def build_checkpoint_state_dict(model, args):
    if args.encoder_usage_info != 'CLIP':
        return model.state_dict()

    clean_clip, _ = openai_clip.load(
        args.pretrained_encoder, device='cpu', jit=False
    )
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in clean_clip.state_dict().items()
    }
    state_dict.update({
        f'visual.{name}': tensor.detach().cpu()
        for name, tensor in model.visual.state_dict().items()
    })
    return state_dict
def extract_visual_state_dict(state_dict):
    visual_state = {
        name[len('visual.'):]: tensor
        for name, tensor in state_dict.items()
        if name.startswith('visual.')
    }
    return visual_state or state_dict


def save_checkpoint(model, optimizer, scaler, epoch, train_loss, args):
    results_dir = Path(args.results_dir)
    epoch_path = results_dir / f'model_{epoch}.pth'
    temporary_path = results_dir / f'.model_{epoch}.pth.tmp'
    checkpoint = {
        'epoch': epoch,
        'state_dict': build_checkpoint_state_dict(model, args),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'train_loss': train_loss,
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all(),
        },
        'attack': {
            'name': 'BadEncoder',
            'target_label': args.target_label,
            'target_word': args.target_word,
            'trigger_file': args.trigger_file,
        },
        'config': vars(args),
    }
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, epoch_path)

    latest_path = results_dir / 'latest.pth'
    latest_temporary = results_dir / '.latest.pth.tmp'
    if latest_temporary.exists() or latest_temporary.is_symlink():
        latest_temporary.unlink()
    try:
        latest_temporary.symlink_to(epoch_path.name)
        os.replace(latest_temporary, latest_path)
    except OSError:
        if latest_temporary.exists() or latest_temporary.is_symlink():
            latest_temporary.unlink()
        shutil.copy2(epoch_path, latest_path)
    print(f'Saved checkpoint: {epoch_path}', flush=True)






def train(
    backdoored_encoder,
    clean_encoder,
    data_loader,
    train_optimizer,
    scaler,
    epoch,
    args,
):
    backdoored_encoder.train()
    for module in backdoored_encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()
    clean_encoder.eval()

    amp_enabled = not args.disable_amp
    total_loss = total_loss_0 = total_loss_1 = total_loss_2 = 0.0
    total_num = 0
    train_bar = tqdm(
        data_loader,
        desc=f'Epoch {epoch}/{args.epochs}',
        dynamic_ncols=True,
        mininterval=1.0,
    )

    for img_clean, img_backdoor_list, reference_list, reference_aug_list in train_bar:
        img_clean = img_clean.cuda(non_blocking=True)
        img_backdoor_list = [
            image.cuda(non_blocking=True) for image in img_backdoor_list
        ]
        reference_list = [
            image.cuda(non_blocking=True) for image in reference_list
        ]
        reference_aug_list = [
            image.cuda(non_blocking=True) for image in reference_aug_list
        ]
        num_references = len(reference_list)
        train_optimizer.zero_grad(set_to_none=True)

        loss_0_value = 0.0
        for img_backdoor, img_reference in zip(
            img_backdoor_list, reference_list
        ):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                feature_backdoor = F.normalize(
                    backdoored_encoder(img_backdoor), dim=-1
                )
                feature_reference = F.normalize(
                    backdoored_encoder(img_reference), dim=-1
                )
                pair_loss = -torch.sum(
                    feature_backdoor * feature_reference, dim=-1
                ).mean()
            scaler.scale(pair_loss / num_references).backward()
            loss_0_value += pair_loss.detach().float().item() / num_references

        loss_1_value = 0.0
        for img_reference, img_reference_aug in zip(
            reference_list, reference_aug_list
        ):
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                clean_feature_reference = F.normalize(
                    clean_encoder(img_reference), dim=-1
                )
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                feature_reference_aug = F.normalize(
                    backdoored_encoder(img_reference_aug), dim=-1
                )
                pair_loss = -torch.sum(
                    feature_reference_aug * clean_feature_reference, dim=-1
                ).mean()
            scaler.scale(
                args.lambda1 * pair_loss / num_references
            ).backward()
            loss_1_value += pair_loss.detach().float().item() / num_references

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            clean_feature_raw = F.normalize(clean_encoder(img_clean), dim=-1)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            feature_raw = F.normalize(backdoored_encoder(img_clean), dim=-1)
            loss_2 = -torch.sum(
                feature_raw * clean_feature_raw, dim=-1
            ).mean()
        scaler.scale(args.lambda2 * loss_2).backward()
        scaler.step(train_optimizer)
        scaler.update()

        batch_size = img_clean.size(0)
        loss_2_value = loss_2.detach().float().item()
        loss_value = (
            loss_0_value
            + args.lambda1 * loss_1_value
            + args.lambda2 * loss_2_value
        )
        total_num += batch_size
        total_loss += loss_value * batch_size
        total_loss_0 += loss_0_value * batch_size
        total_loss_1 += loss_1_value * batch_size
        total_loss_2 += loss_2_value * batch_size
        train_bar.set_postfix(
            loss=f'{total_loss / total_num:.4f}',
            loss0=f'{total_loss_0 / total_num:.4f}',
            loss1=f'{total_loss_1 / total_num:.4f}',
            loss2=f'{total_loss_2 / total_num:.4f}',
            lr=f'{train_optimizer.param_groups[0]["lr"]:.1e}',
        )

    return total_loss / total_num



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finetune the encoder to get the backdoored encoder')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lr', default=1e-6, type=float, help='learning rate in SGD')
    parser.add_argument('--lambda1', default=1.0, type=np.float64, help='value of labmda1')
    parser.add_argument('--lambda2', default=1.0, type=np.float64, help='value of labmda2')
    parser.add_argument('--epochs', default=10, type=int, help='Number of sweeps over the shadow dataset to inject the backdoor')

    parser.add_argument('--reference_file', default='', type=str, help='path to the reference inputs')
    parser.add_argument('--trigger_file', default='', type=str, help='path to the trigger')
    parser.add_argument('--shadow_dataset', default='cc3m', type=str, help='shadow dataset')
    parser.add_argument('--pretrained_encoder', default='', type=str, help='path to the clean encoder used to finetune the backdoored encoder')
    parser.add_argument('--encoder_usage_info', default='CLIP', type=str, help='encoder usage information')
    parser.add_argument('--shadow_data_root', default='', type=str, help='root containing shadow dataset images')
    parser.add_argument('--shadow_csv', default='', type=str, help='CSV manifest for the shadow dataset')
    parser.add_argument('--max_shadow_samples', default=None, type=int, help='optional shadow subset size for smoke tests')
    parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--target_label', default=954, type=int, help='zero-shot target class index')
    parser.add_argument('--target_word', default='banana', type=str, help='zero-shot target class name')
    parser.add_argument('--disable_amp', action='store_true', help='disable mixed precision training')
    parser.add_argument('--resume', default='', type=str, help='checkpoint used to resume training')
    parser.add_argument('--checkpoint_every', default=1, type=int, help='save a checkpoint every N epochs')

    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save the backdoored encoder')

    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')
    args = parser.parse_args()
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)

    # Set the seed and determine the GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Specify the pre-training data directory
    args.data_dir = f'./data/{args.shadow_dataset.split("_")[0]}/'
    args.knn_k = 200
    args.knn_t = 0.5
    args.reference_label = args.target_label
    print(args, flush=True)

    # Create the Pytorch Datasets, and create the data loader for the training set
    # memory_data, test_data_clean, and test_data_backdoor are used to monitor the finetuning process. They are not reqruied by our BadEncoder
    shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset(args)
    train_loader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    clean_model = get_encoder_architecture_usage(args).cuda()
    model = get_encoder_architecture_usage(args).cuda()

    if args.encoder_usage_info != 'CLIP':
        raise ValueError('This entry point supports the CLIP baseline only.')
    if not args.pretrained_encoder:
        raise ValueError('--pretrained_encoder is required')
    if args.checkpoint_every < 1:
        raise ValueError('--checkpoint_every must be at least 1')

    print('Optimizer: SGD', flush=True)
    optimizer = torch.optim.SGD(
        model.visual.parameters(),
        lr=args.lr,
        weight_decay=5e-4,
        momentum=0.9,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=not args.disable_amp)

    print(f'Loading clean model from {args.pretrained_encoder}', flush=True)
    clean_checkpoint = torch.jit.load(
        args.pretrained_encoder, map_location='cpu'
    )
    clean_state_dict = clean_checkpoint.visual.state_dict()
    clean_model.visual.load_state_dict(clean_state_dict)
    model.visual.load_state_dict(clean_state_dict)
    del clean_checkpoint, clean_state_dict

    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
        checkpoint = torch.load(
            resume_path, map_location='cpu', weights_only=False
        )
        model.visual.load_state_dict(
            extract_visual_state_dict(checkpoint['state_dict'])
        )
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scaler'):
            scaler.load_state_dict(checkpoint['scaler'])
        rng_state = checkpoint.get('rng_state')
        if rng_state:
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'])
            torch.cuda.set_rng_state_all(rng_state['cuda'])
        start_epoch = int(checkpoint['epoch']) + 1
        print(
            f'Resuming from {resume_path} at epoch {start_epoch}',
            flush=True,
        )
        del checkpoint

    if start_epoch > args.epochs:
        print(
            f'Checkpoint already reached epoch {start_epoch - 1}; '
            f'nothing to train for --epochs {args.epochs}.',
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        print('=' * 60, flush=True)
        train_loss = train(
            model.visual,
            clean_model.visual,
            train_loader,
            optimizer,
            scaler,
            epoch,
            args,
        )
        print(
            f'Epoch {epoch}/{args.epochs} finished: loss={train_loss:.6f}',
            flush=True,
        )
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(
                model, optimizer, scaler, epoch, train_loss, args
            )
        torch.cuda.empty_cache()
