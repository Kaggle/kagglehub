"""Simple constant-velocity tracker used for gating/linking across frames.

This is a lightweight, deterministic tracker (not a full probabilistic Kalman filter).
It predicts next positions using a constant-velocity estimate from the last two observations
and gates candidate links by distance. Optionally combines appearance cost.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def _to_coord_microns(n: Dict, voxel_size: Tuple[float, float, float]):
    vz, vy, vx = voxel_size
    return np.array([n['z'] * vz, n['y'] * vy, n['x'] * vx], dtype=float)


def link_nodes_with_kalman(nodes: List[Dict], images: Optional[np.ndarray] = None, max_dist_um: float = 7.0,
                           voxel_size: Tuple[float,float,float]=(1.625,0.40625,0.40625),
                           appearance_weight: float = 0.0, sample_radius: int = 2) -> List[Dict]:
    """Sequential linking across time using constant-velocity predictions.

    Returns list of edge dicts {'source_id', 'target_id'}.
    """
    if linear_sum_assignment is None:
        raise RuntimeError('scipy is required for kalman linking')
    # group nodes by time
    nodes_by_t = {}
    for n in nodes:
        nodes_by_t.setdefault(int(n['t']), []).append(n)
    times = sorted(nodes_by_t.keys())
    if not times:
        return []

    # precompute appearance intensities if images provided
    intensities = {}
    if images is not None:
        for t, nlist in nodes_by_t.items():
            if images.ndim == 5:
                frame = images[t, 0]
            elif images.ndim == 4:
                frame = images[t]
            else:
                raise RuntimeError('images must be 4D or 5D')
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

    # Initialize tracks with nodes from first time
    tracks = []  # each track: {'nodes':[n...], 'last_pos':array, 'vel':array or None}
    for n in nodes_by_t[times[0]]:
        tracks.append({'nodes':[n], 'last_pos':_to_coord_microns(n, voxel_size), 'vel':None})

    # process subsequent times
    for t in times[1:]:
        candidates = nodes_by_t[t]
        if len(candidates) == 0:
            # nothing to assign; keep tracks as ended
            continue
        cand_coords = np.vstack([_to_coord_microns(c, voxel_size) for c in candidates])

        # predict positions for current tracks
        preds = []
        for tr in tracks:
            if tr['vel'] is not None:
                preds.append(tr['last_pos'] + tr['vel'])
            else:
                preds.append(tr['last_pos'])
        if len(preds) == 0:
            # start new tracks from candidates
            for c in candidates:
                tracks.append({'nodes':[c], 'last_pos':_to_coord_microns(c, voxel_size), 'vel':None})
            continue
        pred_coords = np.vstack(preds)

        # distance cost
        dists = np.linalg.norm(pred_coords[:, None, :] - cand_coords[None, :, :], axis=2)

        # appearance cost
        if appearance_weight > 0.0 and images is not None:
            I_pred = np.array([intensities.get((tr['nodes'][-1]['t'], int(tr['nodes'][-1]['node_id'])), 0.0) for tr in tracks])[:, None]
            I_cand = np.array([intensities.get((t, int(c['node_id'])), 0.0) for c in candidates])[None, :]
            idiffs = np.abs(I_pred - I_cand)
            cost = dists + appearance_weight * (idiffs * max_dist_um)
        else:
            cost = dists

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tr = set()
        assigned_cand = set()
        for i, j in zip(row_ind, col_ind):
            if dists[i, j] <= max_dist_um:
                tr = tracks[i]
                c = candidates[j]
                # update track
                tr['nodes'].append(c)
                new_pos = _to_coord_microns(c, voxel_size)
                if tr['vel'] is None and len(tr['nodes']) >= 2:
                    prev_pos = tr['last_pos']
                    tr['vel'] = new_pos - prev_pos
                elif tr['vel'] is not None:
                    prev_pos = tr['last_pos']
                    # simple velocity update (momentum)
                    tr['vel'] = 0.5 * tr['vel'] + 0.5 * (new_pos - prev_pos)
                tr['last_pos'] = new_pos
                assigned_tr.add(i)
                assigned_cand.add(j)

        # unassigned candidates -> new tracks
        for idx, c in enumerate(candidates):
            if idx not in assigned_cand:
                tracks.append({'nodes':[c], 'last_pos':_to_coord_microns(c, voxel_size), 'vel':None})

    # build edges from track consecutive nodes
    edges = []
    for tr in tracks:
        for a, b in zip(tr['nodes'], tr['nodes'][1:]):
            edges.append({'source_id': int(a['node_id']), 'target_id': int(b['node_id'])})
    return edges
