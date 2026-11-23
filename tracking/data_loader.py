"""Simple zarr data loader for 3D+time volumes and label centroids.

Dependencies: zarr, numpy, scipy

Provides:
- open_zarr(path): returns zarr array and metadata (voxel_size)
- iter_frames(zarr_array): yields (t, ndarray) frames
- centroids_from_labels(label_array, voxel_size=(1.0,1.0,1.0)): returns list of (t, z, y, x) centroids in voxel coords and physical coords
"""
from typing import Iterator, List, Tuple, Dict, Optional
import numpy as np
try:
    import zarr
except Exception as e:
    zarr = None

try:
    from scipy.ndimage import center_of_mass
except Exception:
    center_of_mass = None


def open_zarr(path: str):
    if zarr is None:
        raise RuntimeError("zarr is required. Install with `pip install zarr`")
    arr = zarr.open(path, mode="r")
    # try to read voxel size from attrs if present
    attrs = getattr(arr, "attrs", {})
    voxel_size = attrs.get("voxel_size", attrs.get("spacing", None))
    return arr, voxel_size


def iter_frames(arr) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield time-index and a 3D ndarray. Supports arrays with shape (T, Z, Y, X)
    or (Z, Y, X) for single timepoint.
    """
    shape = getattr(arr, "shape", None)
    if shape is None:
        raise RuntimeError("Provided object does not have shape")
    if len(shape) == 4:
        for t in range(shape[0]):
            yield t, np.asarray(arr[t])
    elif len(shape) == 3:
        yield 0, np.asarray(arr)
    else:
        raise RuntimeError(f"Unsupported array shape: {shape}")


def centroids_from_labels(label_arr: np.ndarray, voxel_size: Optional[Tuple[float, float, float]] = None) -> List[Dict]:
    """Compute centroids per connected label per timepoint.

    label_arr: either 4D (T,Z,Y,X) or 3D (Z,Y,X)
    Returns list of dicts: {"t":int, "z":float, "y":float, "x":float, "z_um":float, "y_um":float, "x_um":float, "label":int}
    """
    if center_of_mass is None:
        raise RuntimeError("scipy is required for centroid extraction. Install with `pip install scipy`")
    out = []
    if label_arr.ndim == 4:
        T = label_arr.shape[0]
        for t in range(T):
            frame = label_arr[t]
            labels = np.unique(frame)
            labels = labels[labels != 0]
            for lab in labels:
                mask = (frame == lab)
                c = center_of_mass(mask)
                if c is None:
                    continue
                z, y, x = c
                entry = {"t": t, "z": z, "y": y, "x": x, "label": int(lab)}
                if voxel_size is not None:
                    vz, vy, vx = voxel_size
                    entry.update({"z_um": z * vz, "y_um": y * vy, "x_um": x * vx})
                out.append(entry)
    elif label_arr.ndim == 3:
        frame = label_arr
        labels = np.unique(frame)
        labels = labels[labels != 0]
        for lab in labels:
            mask = (frame == lab)
            c = center_of_mass(mask)
            if c is None:
                continue
            z, y, x = c
            entry = {"t": 0, "z": z, "y": y, "x": x, "label": int(lab)}
            if voxel_size is not None:
                vz, vy, vx = voxel_size
                entry.update({"z_um": z * vz, "y_um": y * vy, "x_um": x * vx})
            out.append(entry)
    else:
        raise RuntimeError("label array must be 3D or 4D")
    return out
