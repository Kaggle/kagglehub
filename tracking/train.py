"""Training script for baseline 3D U-Net.

Usage (example):
  python3 -m tracking.train --images images.npy --labels labels.npy --epochs 10 --batch-size 1
"""
import argparse
import numpy as np
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', default='model.pth')
    p.add_argument('--checkpoint-interval', type=int, default=0, help='save checkpoint every N epochs (0=disabled)')
    p.add_argument('--amp', action='store_true', help='use automatic mixed precision when CUDA is available')
    return p.parse_args()


def load_array(path):
    if path.endswith('.npz') or path.endswith('.npy'):
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            keys = list(data.keys())
            arr = data[keys[0]]
        else:
            arr = data
        return arr
    else:
        raise RuntimeError('Only .npz/.npy supported for training demo')


def main():
    args = parse_args()
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except Exception:
        raise RuntimeError('PyTorch is required to run training')

    from tracking.unet3d import UNet3D
    from tracking.dataset import VolumesDataset

    images = load_array(args.images)
    labels = load_array(args.labels)
    # Expect shapes: (N,Z,Y,X) or (N,C,Z,Y,X)
    ds = VolumesDataset(images, labels)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_channels=1, out_channels=1, base_filters=16).to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    use_amp = args.amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, lbls in dl:
            imgs = imgs.to(device)
            lbls = lbls.to(device).float()
            opt.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(imgs)
                    if preds.shape != lbls.shape:
                        lbls = lbls.unsqueeze(1)
                    loss = criterion(preds, lbls)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                preds = model(imgs)
                if preds.shape != lbls.shape:
                    lbls = lbls.unsqueeze(1)
                loss = criterion(preds, lbls)
                loss.backward()
                opt.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(dl))
        print(f'Epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f}')

        # checkpoint
        if args.checkpoint_interval > 0 and ((epoch + 1) % args.checkpoint_interval == 0):
            ckpt = {
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
            }
            ckpt_path = f"{os.path.splitext(args.out)[0]}_epoch{epoch+1}.pth"
            torch.save(ckpt, ckpt_path)
            print('Saved checkpoint to', ckpt_path)

    # final save
    torch.save({'model_state': model.state_dict(), 'opt_state': opt.state_dict()}, args.out)
    print('Saved model to', args.out)


if __name__ == '__main__':
    main()
