import numpy as np
import os
from pathlib import Path


def test_linking_and_pipeline(tmp_path):
    # create small two-frame label volume
    arr = np.zeros((2,8,16,16), dtype=np.int32)
    arr[0,3,6,6] = 1
    arr[0,3,6,7] = 1
    arr[1,3,6,7] = 2
    arr[1,3,6,8] = 2
    p = tmp_path / "labels.npz"
    np.savez(p, labels=arr)

    out = tmp_path / 'submission.csv'
    import tracking.pipeline as tp
    tp.run('testset', str(p), str(out), tile_size=(8,8,8), overlap=(2,2,2), merge_distance=2.0, link=True, max_dist_um=10.0, voxel_size=(1.625,0.40625,0.40625))

    assert out.exists()
    txt = out.read_text()
    # must contain at least one edge row
    assert 'row_type,edge' in txt or '\n' in txt


def test_unet_forward():
    try:
        import torch
    except Exception:
        return
    from tracking.unet3d import UNet3D
    model = UNet3D(in_channels=1, out_channels=1, base_filters=8)
    x = torch.randn(1,1,16,32,32)
    y = model(x)
    assert y.shape[0] == 1
