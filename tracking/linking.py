"""Simple frame-to-frame linking using Hungarian assignment on scaled centroid distances.

Functions:
- link_nodes(nodes, max_dist_um=7.0, voxel_size=(1.625,0.40625,0.40625)) -> edges list of dicts {source_id,target_id}

`nodes` is a list of dicts with keys `node_id,t,z,y,x`.
"""
from typing import List, Dict, Tuple
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def _to_coord(n: Dict, voxel_size: Tuple[float, float, float]):
    vz, vy, vx = voxel_size
    return np.array([n['z'] * vz, n['y'] * vy, n['x'] * vx], dtype=float)


def link_nodes(nodes: List[Dict], max_dist_um: float = 7.0, voxel_size: Tuple[float,float,float]=(1.625,0.40625,0.40625)) -> List[Dict]:
    if linear_sum_assignment is None:
        raise RuntimeError('scipy is required for linking. Install with `pip install scipy`')
    # group nodes by time
    nodes_by_t = {}
    for n in nodes:
        nodes_by_t.setdefault(int(n['t']), []).append(n)

    edges = []
    times = sorted(nodes_by_t.keys())
    for t0, t1 in zip(times, times[1:]):
        src_nodes = nodes_by_t[t0]
        tgt_nodes = nodes_by_t[t1]
        if len(src_nodes) == 0 or len(tgt_nodes) == 0:
            continue
        src_coords = np.vstack([_to_coord(n, voxel_size) for n in src_nodes])
        tgt_coords = np.vstack([_to_coord(n, voxel_size) for n in tgt_nodes])
        dists = np.linalg.norm(src_coords[:, None, :] - tgt_coords[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(dists)
        for i, j in zip(row_ind, col_ind):
            if dists[i, j] <= max_dist_um:
                edges.append({'source_id': int(src_nodes[i]['node_id']), 'target_id': int(tgt_nodes[j]['node_id'])})
    return edges


def link_nodes_with_appearance(nodes: List[Dict], images: np.ndarray, max_dist_um: float = 7.0,
                               voxel_size: Tuple[float,float,float]=(1.625,0.40625,0.40625),
                               appearance_weight: float = 1.0, sample_radius: int = 2) -> List[Dict]:
    """Link nodes using a combined cost: distance (um) + appearance_weight * intensity_diff.

    `images` should be a numpy array shaped (T,Z,Y,X) or (T,C,Z,Y,X) with intensity values.
    intensity values are normalized per-frame to [0,1] before computing differences.
    """
    if linear_sum_assignment is None:
        raise RuntimeError('scipy is required for linking. Install with `pip install scipy`')
    # group nodes by time
    nodes_by_t = {}
    for n in nodes:
        nodes_by_t.setdefault(int(n['t']), []).append(n)

    # compute mean intensity per node
    intensities = {}
    for t, nlist in nodes_by_t.items():
        # select frame
        if images.ndim == 5:
            frame = images[t, 0]
        elif images.ndim == 4:
            frame = images[t]
        else:
            raise RuntimeError('images must be 4D or 5D (T,Z,Y,X) or (T,C,Z,Y,X)')
        # normalize to [0,1]
        fmin, fmax = float(frame.min()), float(frame.max())
        den = (fmax - fmin) if (fmax - fmin) > 0 else 1.0
        fnorm = (frame - fmin) / den
        for n in nlist:
            cz, cy, cx = int(round(n['z'])), int(round(n['y'])), int(round(n['x']))
            z0 = max(0, cz - sample_radius)
            z1 = min(frame.shape[0], cz + sample_radius + 1)
            y0 = max(0, cy - sample_radius)
            y1 = min(frame.shape[1], cy + sample_radius + 1)
            x0 = max(0, cx - sample_radius)
            x1 = min(frame.shape[2], cx + sample_radius + 1)
            patch = fnorm[z0:z1, y0:y1, x0:x1]
            intensities[(t, int(n['node_id']))] = float(patch.mean()) if patch.size > 0 else 0.0

    edges = []
    times = sorted(nodes_by_t.keys())
    for t0, t1 in zip(times, times[1:]):
        src_nodes = nodes_by_t[t0]
        tgt_nodes = nodes_by_t[t1]
        if len(src_nodes) == 0 or len(tgt_nodes) == 0:
            continue
        src_coords = np.vstack([_to_coord(n, voxel_size) for n in src_nodes])
        tgt_coords = np.vstack([_to_coord(n, voxel_size) for n in tgt_nodes])
        dists = np.linalg.norm(src_coords[:, None, :] - tgt_coords[None, :, :], axis=2)

        # compute intensity diffs
        I_src = np.array([intensities.get((t0, int(n['node_id'])), 0.0) for n in src_nodes])[:, None]
        I_tgt = np.array([intensities.get((t1, int(n['node_id'])), 0.0) for n in tgt_nodes])[None, :]
        idiffs = np.abs(I_src - I_tgt)

        # combined cost (lower is better)
        cost = dists + appearance_weight * (idiffs * max_dist_um)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            if dists[i, j] <= max_dist_um:
                edges.append({'source_id': int(src_nodes[i]['node_id']), 'target_id': int(tgt_nodes[j]['node_id'])})
    return edges
