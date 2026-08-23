"""Division detection utilities.

Detects candidate cell division events (parent -> two children) based on spatial
proximity in subsequent frames and optional intensity/size heuristics.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.ndimage import binary_dilation


def detect_divisions(nodes: List[Dict], max_child_dist_vox: float = 5.0, min_child_sep_vox: float = 2.0, window: int = 1) -> List[Tuple[Dict, List[Dict]]]:
    """Simple proximity-based division detection (kept for backward compat).
    """
    nodes_by_t = {}
    for n in nodes:
        nodes_by_t.setdefault(int(n['t']), []).append(n)

    divisions = []
    times = sorted(nodes_by_t.keys())
    for t in times:
        parents = nodes_by_t.get(t, [])
        children = nodes_by_t.get(t+1, []) if window >= 1 else []
        if not children:
            continue
        child_coords = np.array([[c['z'], c['y'], c['x']] for c in children])
        for p in parents:
            pcoord = np.array([p['z'], p['y'], p['x']])
            dists = np.linalg.norm((child_coords - pcoord), axis=1)
            idx = np.where(dists <= max_child_dist_vox)[0]
            if idx.size >= 2:
                sel = child_coords[idx]
                pairwise = np.linalg.norm(sel[:, None, :] - sel[None, :, :], axis=2)
                i_max, j_max = np.unravel_index(np.argmax(pairwise), pairwise.shape)
                if pairwise[i_max, j_max] >= min_child_sep_vox:
                    child_nodes = [children[int(idx[i_max])], children[int(idx[j_max])]]
                    divisions.append((p, child_nodes))
    return divisions


def detect_divisions_enhanced(nodes: List[Dict], labels: np.ndarray, images: Optional[np.ndarray] = None,
                              voxel_size=(1.0,1.0,1.0), max_child_dist_vox: float = 5.0,
                              min_child_sep_vox: float = 1.0, min_iou: float = 0.05,
                              volume_ratio_tol: float = 0.6, intensity_ratio_tol: float = 0.6) -> List[Tuple[Dict, List[Dict], float]]:
    """Enhanced division detection using masks and intensity/volume heuristics.

    Returns list of (parent, [child1, child2], score) where score is between 0 and 1.
    """
    # nodes_by_t
    nodes_by_t = {}
    for n in nodes:
        nodes_by_t.setdefault(int(n['t']), []).append(n)

    divisions = []
    times = sorted(nodes_by_t.keys())
    for t in times:
        parents = nodes_by_t.get(t, [])
        children = nodes_by_t.get(t+1, [])
        if not children:
            continue
        child_coords = np.array([[c['z'], c['y'], c['x']] for c in children])
        for p in parents:
            pcoord = np.array([p['z'], p['y'], p['x']])
            # find candidate child labels by scanning the next-frame label volume
            zc, yc, xc = int(round(p['z'])), int(round(p['y'])), int(round(p['x']))
            z0 = max(0, zc - int(max_child_dist_vox))
            z1 = min(labels.shape[1]-1, zc + int(max_child_dist_vox))
            y0 = max(0, yc - int(max_child_dist_vox))
            y1 = min(labels.shape[2]-1, yc + int(max_child_dist_vox))
            x0 = max(0, xc - int(max_child_dist_vox))
            x1 = min(labels.shape[3]-1, xc + int(max_child_dist_vox))
            local = labels[int(t+1), z0:z1+1, y0:y1+1, x0:x1+1]
            cand_labels = np.unique(local)
            cand_labels = cand_labels[(cand_labels != 0)]
            if cand_labels.size < 2:
                continue
            # build child node dicts by locating centroids for candidate labels
            label_to_node = {}
            for lab in cand_labels:
                mask = (labels[int(t+1)] == int(lab))
                # compute centroid
                coords = np.array(np.where(mask))
                if coords.size == 0:
                    continue
                meanz, meany, meanx = coords.mean(axis=1)
                label_to_node[int(lab)] = {'node_id': int(lab), 't': int(t+1), 'z': float(meanz), 'y': float(meany), 'x': float(meanx)}
            cand_list = list(label_to_node.values())
            if len(cand_list) < 2:
                continue
            # evaluate all pairs among candidate child masks
            best = None
            for i in range(len(cand_list)):
                for j in range(i+1, len(cand_list)):
                    ci = cand_list[i]
                    cj = cand_list[j]
                    # separation
                    sep = np.linalg.norm(np.array([ci['z'],ci['y'],ci['x']]) - np.array([cj['z'],cj['y'],cj['x']]))
                    if sep < min_child_sep_vox:
                        continue
                    # map centroids to nearest integer voxel to find label ids
                    pz, py, px = int(round(p['z'])), int(round(p['y'])), int(round(p['x']))
                    iz, iy, ix = int(round(ci['z'])), int(round(ci['y'])), int(round(ci['x']))
                    jz, jy, jx = int(round(cj['z'])), int(round(cj['y'])), int(round(cj['x']))
                    try:
                        parent_label = int(labels[int(t), int(round(p['z'])), int(round(p['y'])), int(round(p['x']))])
                    except Exception:
                        parent_label = 0
                    # child labels inferred from candidate dicts (node_id set to label)
                    child_label_i = int(ci['node_id'])
                    child_label_j = int(cj['node_id'])
                    if parent_label == 0 or child_label_i == 0 or child_label_j == 0:
                        continue
                    parent_mask = (labels[int(t)] == parent_label)
                    child_mask_i = (labels[int(t+1)] == child_label_i)
                    child_mask_j = (labels[int(t+1)] == child_label_j)
                    # volumes
                    vol_p = float(parent_mask.sum())
                    vol_csum = float(child_mask_i.sum() + child_mask_j.sum())
                    vol_ratio = vol_csum / max(1.0, vol_p)
                    # intensity means
                    if images is not None:
                        ip = float(images[int(t)][parent_mask].mean()) if parent_mask.any() else 0.0
                        ic = float(np.concatenate([images[int(t+1)][child_mask_i].ravel(), images[int(t+1)][child_mask_j].ravel()]).mean()) if (child_mask_i.any() or child_mask_j.any()) else 0.0
                        intensity_ratio = (ic / max(1e-6, ip)) if ip > 0 else 1.0
                    else:
                        intensity_ratio = 1.0
                    # IoU between parent mask and union of children masks (across frames)
                    union_children = child_mask_i | child_mask_j
                    # dilate parent mask to allow small motion
                    pm_dil = binary_dilation(parent_mask, iterations=1)
                    inter = float((pm_dil & union_children).sum())
                    union = float((pm_dil | union_children).sum())
                    iou = inter / union if union > 0 else 0.0
                    # score components
                    score_vol = max(0.0, 1.0 - abs(np.log(max(1e-6, vol_ratio))))
                    score_int = max(0.0, 1.0 - abs(np.log(max(1e-6, intensity_ratio))))
                    score_iou = iou
                    # combine (weighted)
                    score = 0.4 * score_iou + 0.3 * score_vol + 0.3 * score_int
                    # thresholds
                    if iou >= min_iou and (vol_ratio >= volume_ratio_tol) and (abs(np.log(max(1e-6, intensity_ratio))) <= np.log(1.0/intensity_ratio_tol)):
                        if best is None or score > best[1]:
                            best = ( (ci, cj), score )
            if best is not None:
                (ci, cj), score = best
                divisions.append((p, [ci, cj], float(score)))
    return divisions
