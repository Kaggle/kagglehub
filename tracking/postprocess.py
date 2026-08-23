"""Postprocessing utilities: extract centroids from tiled label outputs and deduplicate to global nodes.

Functions:
- extract_nodes_from_label_tiles(label_arr, tile_size, overlap, merge_distance_voxels)

Assumes `label_arr` is a 4D numpy array (T,Z,Y,X) of integer labels or 3D (Z,Y,X).
"""
from typing import List, Dict, Tuple
import numpy as np

try:
    from scipy.ndimage import center_of_mass
except Exception:
    center_of_mass = None

from .io import tiled_reader


def _euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def extract_nodes_from_label_tiles(label_arr: np.ndarray, tile_size=(64,64,64), overlap=(16,16,16), merge_distance_voxels: float = 3.0) -> List[Dict]:
    """Extract centroids from tiled label array and deduplicate per timepoint.

    Returns list of nodes: dicts with `node_id`, `t`, `z`, `y`, `x` (voxel coords).
    """
    if center_of_mass is None:
        raise RuntimeError("scipy is required for centroid extraction. Install with `pip install scipy`")
    all_nodes = []
    next_node_id = 1
    # collect per-timepoint centroids
    centroids_by_t = {}
    for t, (z0z1), (y0y1), (x0x1), tile in tiled_reader(label_arr, tile_size=tile_size, overlap=overlap):
        z0, z1 = z0z1
        y0, y1 = y0y1
        x0, x1 = x0x1
        labels = np.unique(tile)
        labels = labels[labels != 0]
        for lab in labels:
            mask = (tile == lab)
            c = center_of_mass(mask)
            if c is None:
                continue
            cz, cy, cx = c
            gz = z0 + cz
            gy = y0 + cy
            gx = x0 + cx
            cent = (gz, gy, gx)
            centroids_by_t.setdefault(t, []).append(cent)

    # deduplicate and assign node ids
    nodes = []
    for t, cents in centroids_by_t.items():
        kept = []  # list of (gz,gy,gx, node_id)
        for c in cents:
            merged = False
            for i, (kgz, kgy, kgx, kid) in enumerate(kept):
                if _euclidean((kgz,kgy,kgx), c) <= merge_distance_voxels:
                    # average coordinates
                    new = ((kgz + c[0]) / 2.0, (kgy + c[1]) / 2.0, (kgx + c[2]) / 2.0)
                    kept[i] = (new[0], new[1], new[2], kid)
                    merged = True
                    break
            if not merged:
                kid = next_node_id
                next_node_id += 1
                kept.append((c[0], c[1], c[2], kid))
        for gz, gy, gx, kid in kept:
            nodes.append({"node_id": int(kid), "t": int(t), "z": float(gz), "y": float(gy), "x": float(gx)})

    return nodes
