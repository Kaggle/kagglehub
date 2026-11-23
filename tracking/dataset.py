"""Simple PyTorch Dataset for 3D volumes and labels stored as numpy arrays or zarr.

Expects inputs to be small enough to fit in memory for this baseline.
"""
from typing import Optional
try:
    import torch
    from torch.utils.data import Dataset
except Exception:
    torch = None

import numpy as np


class VolumesDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: Optional[np.ndarray] = None, transform=None):
        """images: (N, C, Z, Y, X) or (N, Z, Y, X) -- will add channel dim if missing
        labels: (N, 1, Z, Y, X) or (N, Z, Y, X)
        """
        if torch is None:
            raise RuntimeError('PyTorch is required to use VolumesDataset')
        self.images = images
        if images.ndim == 4:
            # (N,Z,Y,X) -> (N,1,Z,Y,X)
            self.images = images[:, None, ...]
        self.labels = labels
        if labels is not None and labels.ndim == 4:
            self.labels = labels[:, None, ...]
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32)
        lbl = None
        if self.labels is not None:
            lbl = self.labels[idx].astype(np.int64)
        if self.transform is not None:
            img, lbl = self.transform(img, lbl)
        import torch
        img_t = torch.from_numpy(img)
        lbl_t = torch.from_numpy(lbl) if lbl is not None else None
        return img_t, lbl_t
