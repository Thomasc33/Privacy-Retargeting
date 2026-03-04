"""
SGN Training Script — Cross-View (CV) Split

Trains SGN models for action recognition or re-identification using the
NTU RGB+D Cross-View benchmark protocol (train: cameras 2,3 / test: camera 1).

Usage:
    python SGN/train.py --task action --dataset ntu60 --data-path NTU/X.pkl
    python SGN/train.py --task privacy --dataset ntu120 --data-path NTU/X.pkl
"""

import argparse
import csv
import os
import os.path as osp
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

# Add parent directory so we can import SGN modules when run as script
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from SGN.model import SGN
from SGN.data import NTUDataLoaders, AverageMeter

# ── Constants ────────────────────────────────────────────────────────────────

TRAIN_CAMERAS = [2, 3]
TEST_CAMERAS = [1]

DATASET_CONFIG = {
    'ntu60': {'action_classes': 60, 'privacy_classes': 40, 'max_action': 60},
    'ntu120': {'action_classes': 120, 'privacy_classes': 106, 'max_action': 120},
}

DEFAULT_HYPERPARAMS = {
    'batch_size': 64,
    'max_epochs': 120,
    'lr': 0.001,
    'weight_decay': 0.0001,
    'lr_milestones': [60, 90, 110],
    'lr_gamma': 0.1,
    'seg': 20,
    'label_smoothing': 0.1,
    'print_freq': 20,
}


# ── NTU Filename Parsing ────────────────────────────────────────────────────

def parse_ntu_filename(filename):
    """Parse NTU RGB+D filename into components.

    Format: SsssCcccPpppRrrrAaaa (e.g. S001C001P001R001A001)
    With SGN 'b' prefix: bSsssCcccPpppRrrrAaaa
    """
    s = str(filename)
    if s.startswith('b'):
        return {
            'S': int(s[3:6]), 'C': int(s[7:10]),
            'P': int(s[11:14]), 'R': int(s[15:18]), 'A': int(s[19:22]),
        }
    return {
        'S': int(s[1:4]), 'C': int(s[5:8]),
        'P': int(s[9:12]), 'R': int(s[13:16]), 'A': int(s[17:20]),
    }


# ── Data Loading with CV Split ──────────────────────────────────────────────

def load_and_split_cv(data_path, task, dataset='ntu60'):
    """Load NTU data and split by Cross-View protocol.

    Args:
        data_path: Path to X.pkl (dict keyed by NTU filenames).
        task: 'action' or 'privacy'.
        dataset: 'ntu60' or 'ntu120'.

    Returns:
        (train_x, train_y, val_x, val_y, test_x, test_y, num_classes)
    """
    import pickle

    cfg = DATASET_CONFIG[dataset]
    num_classes = cfg['action_classes'] if task == 'action' else cfg['privacy_classes']

    with open(data_path, 'rb') as f:
        X = pickle.load(f)

    # Filter to dataset scope
    if dataset == 'ntu60':
        X = {k: v for k, v in X.items() if parse_ntu_filename(k)['A'] <= 60}

    # Split by camera view
    train_keys, test_keys = [], []
    for key in X:
        info = parse_ntu_filename(key)
        if info['C'] in TRAIN_CAMERAS:
            train_keys.append(key)
        elif info['C'] in TEST_CAMERAS:
            test_keys.append(key)

    # Build arrays  (shape: N x 300 x 150)
    def build_arrays(keys):
        x = np.zeros((len(keys), 300, 150), dtype=np.float32)
        y = np.zeros(len(keys), dtype=np.int64)
        for i, key in enumerate(keys):
            data = X[key]
            if hasattr(data, 'numpy'):
                data = data.numpy()
            data = data[:, :, :3]  # keep only xyz
            num_t = min(data.shape[0], 300)
            x[i, :num_t, :75] = data[:num_t].reshape(num_t, -1)

            info = parse_ntu_filename(key)
            if task == 'action':
                y[i] = info['A'] - 1
            else:  # privacy / re-identification
                y[i] = info['P'] - 1
            # Clamp class index to valid range
            y[i] = min(y[i], num_classes - 1)
        return x, y

    train_x, train_y = build_arrays(train_keys)
    test_x, test_y = build_arrays(test_keys)

    # Hold out 15% of training data as validation
    n_val = max(1, int(len(train_x) * 0.15))
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(train_x))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    val_x, val_y = train_x[val_indices], train_y[val_indices]
    train_x, train_y = train_x[train_indices], train_y[train_indices]

    print(f"CV split — train: {len(train_x)}, val: {len(val_x)}, test: {len(test_x)}")
    print(f"Task: {task}, classes: {num_classes}")

    return train_x, train_y, val_x, val_y, test_x, test_y, num_classes


# ── Training Utilities ──────────────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return correct.view(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size)


def train_epoch(loader, model, criterion, optimizer, epoch, print_freq=20):
    losses = AverageMeter()
    acces = AverageMeter()
    model.train()

    for i, (inputs, target) in enumerate(loader):
        output = model(inputs.cuda())
        target = target.cuda()
        loss = criterion(output, target)

        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % print_freq == 0:
            print(f'  Epoch {epoch+1:>3d}  batch {i+1:>3d}  '
                  f'loss {losses.val:.4f} ({losses.avg:.4f})  '
                  f'acc {acces.val:.1f} ({acces.avg:.1f})')

    return losses.avg, acces.avg


def validate_epoch(loader, model, criterion):
    losses = AverageMeter()
    acces = AverageMeter()
    model.eval()

    for inputs, target in loader:
        with torch.no_grad():
            output = model(inputs.cuda())
            target = target.cuda()
            loss = criterion(output, target)

        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

    return losses.avg, acces.avg


# ── Main Training Function ──────────────────────────────────────────────────

def model_output_path(task, dataset):
    """Return the standard output path for a trained SGN model."""
    ds_suffix = '60' if dataset == 'ntu60' else '120'
    return osp.join('SGN', 'pretrained', f'{task}_{ds_suffix}.pt')


def train_sgn_model(task, dataset='ntu60', data_path='NTU/X.pkl',
                    device='cuda:0', output_path=None):
    """Train an SGN model for a given task and dataset.

    Args:
        task: 'action' or 'privacy'
        dataset: 'ntu60' or 'ntu120'
        data_path: Path to NTU X.pkl data file.
        device: Torch device string.
        output_path: Override output checkpoint path.

    Returns:
        Path to the saved checkpoint.
    """
    hp = DEFAULT_HYPERPARAMS
    if output_path is None:
        output_path = model_output_path(task, dataset)

    os.makedirs(osp.dirname(output_path), exist_ok=True)

    # Load data with CV split
    train_x, train_y, val_x, val_y, test_x, test_y, num_classes = \
        load_and_split_cv(data_path, task, dataset)

    # Data loaders
    loaders = NTUDataLoaders(
        dataset='NTU', case=0, seg=hp['seg'],
        train_X=train_x, train_Y=train_y,
        val_X=val_x, val_Y=val_y,
        test_X=test_x, test_Y=test_y,
        aug=0,
    )
    train_loader = loaders.get_train_loader(hp['batch_size'], 0)
    val_loader = loaders.get_val_loader(hp['batch_size'], 0)

    # Model
    model = SGN(num_classes, 'NTU', hp['seg'], hp['batch_size'], 1)
    if torch.cuda.is_available():
        model = model.cuda()
    print(f'SGN parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = LabelSmoothingLoss(num_classes, smoothing=hp['label_smoothing']).cuda()
    optimizer = optim.Adam(model.parameters(), lr=hp['lr'],
                           weight_decay=hp['weight_decay'])
    scheduler = MultiStepLR(optimizer, milestones=hp['lr_milestones'],
                            gamma=hp['lr_gamma'])

    best_acc = -np.inf
    best_epoch = 0
    log = []

    for epoch in range(hp['max_epochs']):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            train_loader, model, criterion, optimizer, epoch, hp['print_freq'])
        val_loss, val_acc = validate_epoch(val_loader, model, criterion)

        log.append([train_loss, float(train_acc.cpu()),
                     val_loss, float(val_acc.cpu())])

        elapsed = time.time() - t0
        print(f'Epoch {epoch+1:>3d} ({elapsed:.1f}s)  '
              f'train loss={train_loss:.4f} acc={train_acc:.1f}  '
              f'val loss={val_loss:.4f} acc={val_acc:.1f}')

        val_acc_scalar = val_acc.cpu() if hasattr(val_acc, 'cpu') else val_acc
        if val_acc_scalar > best_acc:
            print(f'  -> New best val_acc: {best_acc:.2f} -> {val_acc_scalar:.2f}, saving to {output_path}')
            best_acc = val_acc_scalar
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': float(best_acc),
                'task': task,
                'dataset': dataset,
                'optimizer': optimizer.state_dict(),
            }, output_path)

        scheduler.step()

    print(f'Training complete. Best val_acc: {best_acc:.2f} at epoch {best_epoch}')
    print(f'Model saved to: {output_path}')

    # Save training log
    log_path = output_path.replace('.pt', '_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['train_loss', 'train_acc', 'val_loss', 'val_acc'])
        writer.writerows(log)

    return output_path


def ensure_sgn_models(config, device='cuda:0'):
    """Check that required SGN models exist; train any that are missing.

    Args:
        config: A Config instance from configs.default_config.
        device: Torch device string.

    Returns:
        dict mapping model name to path.
    """
    data_path = config.data.data_path
    dataset = 'ntu120' if config.data.dataset_name == 'NTU-120' else 'ntu60'

    required = {
        'action': config.evaluation.sgn_action_model_path,
        'privacy': config.evaluation.sgn_reid_model_path,
    }

    for task, path in required.items():
        if not osp.exists(path):
            print(f'SGN model not found: {path}')
            if not osp.exists(data_path):
                raise FileNotFoundError(
                    f'Cannot train SGN model — data file not found: {data_path}\n'
                    f'Please provide the NTU dataset or a pre-trained SGN model at {path}'
                )
            print(f'Training SGN {task} model ({dataset})...')
            train_sgn_model(task, dataset, data_path, device, output_path=path)
        else:
            print(f'SGN {task} model found: {path}')

    return required


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train SGN models (Cross-View split)')
    parser.add_argument('--task', choices=['action', 'privacy'], required=True,
                        help='Training task: action recognition or privacy/re-id')
    parser.add_argument('--dataset', choices=['ntu60', 'ntu120'], default='ntu60',
                        help='Dataset variant')
    parser.add_argument('--data-path', default='NTU/X.pkl',
                        help='Path to NTU X.pkl data file')
    parser.add_argument('--device', default='cuda:0',
                        help='Torch device')
    parser.add_argument('--output', default=None,
                        help='Override output path for checkpoint')
    args = parser.parse_args()

    train_sgn_model(
        task=args.task,
        dataset=args.dataset,
        data_path=args.data_path,
        device=args.device,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
