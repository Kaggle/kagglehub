"""Tiled IO helpers and optional Dask-backed zarr loader.

Functions:
- tiled_reader(arr, tile_size=(64,64,64), overlap=(16,16,16)) -> yields (t, (z0,z1),(y0,y1),(x0,x1), tile_array)
- dask_open_zarr(path) -> dask.array
"""
from typing import Tuple, Iterator
import numpy as np

try:
    import dask.array as da
except Exception:
    da = None


def _slice_ranges(length: int, size: int, overlap: int):
    """Yield (start, end) ranges covering [0,length) with given tile size and overlap."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0:
        overlap = 0
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than size")
    starts = list(range(0, max(1, length - overlap), step))
    ranges = []
    for s in starts:
        e = min(length, s + size)
        ranges.append((s, e))
    # ensure last tile reaches end
    if ranges and ranges[-1][1] < length:
        s = max(0, length - size)
        ranges.append((s, length))
    return ranges


def tiled_reader(arr: np.ndarray, tile_size=(64, 64, 64), overlap=(16, 16, 16)) -> Iterator[Tuple[int, Tuple[int,int], Tuple[int,int], Tuple[int,int], np.ndarray]]:
    """Yield tiles for a 4D array of shape (T,Z,Y,X) or a 3D array (Z,Y,X).

    Yields: (t, (z0,z1), (y0,y1), (x0,x1), tile_array)
    """
    if arr.ndim == 4:
        T, Z, Y, X = arr.shape
    elif arr.ndim == 3:
        T = 1
        Z, Y, X = arr.shape
        arr = arr[None, ...]
    else:
        raise ValueError("arr must be 3D or 4D")
    tz, ty, tx = tile_size
    oz, oy, ox = overlap
    z_ranges = _slice_ranges(Z, tz, oz)
    y_ranges = _slice_ranges(Y, ty, oy)
    x_ranges = _slice_ranges(X, tx, ox)
    for t in range(T):
        for zr in z_ranges:
            for yr in y_ranges:
                for xr in x_ranges:
                    z0, z1 = zr
                    y0, y1 = yr
                    x0, x1 = xr
                    tile = np.asarray(arr[t, z0:z1, y0:y1, x0:x1])
                    yield t, (z0, z1), (y0, y1), (x0, x1), tile


def dask_open_zarr(path: str):
    """Return a Dask array backed by a zarr store. Requires dask.
    Example: darr = dask_open_zarr('path/to/store.zarr')
    """
    if da is None:
        raise RuntimeError("dask is required for dask_open_zarr. Install with `pip install dask[array]`")
    return da.from_zarr(path)
