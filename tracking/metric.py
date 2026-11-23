"""Simplified implementation of the competition metric for local evaluation.

This implements per-timepoint node matching (Hungarian) with a max distance threshold
and computes an edge Jaccard and a simple division Jaccard based on out-degree splits.

Dependencies: numpy, scipy
"""
from typing import List, Tuple, Dict
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def match_nodes(gt: List[Dict], pred: List[Dict], voxel_size=(1.0, 0.40625, 0.40625), max_dist_um=7.0):
    """Match nodes for a single timepoint. Expects lists of dicts with keys 'z','y','x'.
    Returns list of (gt_idx, pred_idx) matches.
    """
    if linear_sum_assignment is None:
        raise RuntimeError("scipy is required for matching. Install with `pip install scipy`")
    if len(gt) == 0 or len(pred) == 0:
        return []
    gt_coords = np.array([[g['z']*voxel_size[0], g['y']*voxel_size[1], g['x']*voxel_size[2]] for g in gt])
    pred_coords = np.array([[p['z']*voxel_size[0], p['y']*voxel_size[1], p['x']*voxel_size[2]] for p in pred])
    dists = np.linalg.norm(gt_coords[:, None, :] - pred_coords[None, :, :], axis=2)
    cost = dists.copy()
    gt_idx, pred_idx = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(gt_idx, pred_idx):
        if dists[i, j] <= max_dist_um:
            matches.append((i, j))
    return matches


def edge_jaccard(gt_nodes_by_t: Dict[int, List[Dict]], gt_edges: List[Tuple[int, int]],
                 pred_nodes_by_t: Dict[int, List[Dict]], pred_edges: List[Tuple[int, int]],
                 voxel_size=(1.0, 0.40625, 0.40625), max_dist_um=7.0):
    """Compute a simplified edge Jaccard across timepoints.

    gt_edges/pred_edges are lists of (source_node_id, target_node_id) where node ids refer to indices
    into the combined node lists; for simplicity this helper assumes node ids are local per-timepoint
    and that nodes_by_t lists are in the same order used by edges. For robust use, convert ids to indices.
    """
    # Build match maps per timepoint
    matched_pred_ids = {}
    gt_to_pred = {}
    for t, gt_nodes in gt_nodes_by_t.items():
        pred_nodes = pred_nodes_by_t.get(t, [])
        matches = match_nodes(gt_nodes, pred_nodes, voxel_size=voxel_size, max_dist_um=max_dist_um)
        for gi, pj in matches:
            gt_to_pred[(t, gi)] = (t, pj)
            matched_pred_ids[(t, pj)] = (t, gi)

    # Convert gt edges to matched pred edges
    gt_edge_set = set()
    for s, e in gt_edges:
        # s and e are (t, idx) tuples expected
        m1 = gt_to_pred.get(tuple(s))
        m2 = gt_to_pred.get(tuple(e))
        if m1 is not None and m2 is not None:
            gt_edge_set.add((m1, m2))

    pred_edge_set = set()
    for s, e in pred_edges:
        pred_edge_set.add((tuple(s), tuple(e)))

    TP = len(gt_edge_set & pred_edge_set)
    FP = len(pred_edge_set - gt_edge_set)
    FN = len(gt_edge_set - pred_edge_set)
    jaccard = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    return {"TP": TP, "FP": FP, "FN": FN, "jaccard": jaccard}


def division_jaccard(gt_edges: List[Tuple[Tuple[int,int], List[Tuple[int,int]]]], pred_edges: List[Tuple[Tuple[int,int], List[Tuple[int,int]]]]):
    """Simplified division Jaccard.

    Here gt_edges is expected as list of tuples: (parent_node, [child_nodes...]) where nodes are (t, idx).
    This function computes TP/FP/FN micro-averaged over all samples.
    """
    gt_divs = set()
    for parent, children in gt_edges:
        gt_divs.add((tuple(parent), tuple(sorted(children))))
    pred_divs = set()
    for parent, children in pred_edges:
        pred_divs.add((tuple(parent), tuple(sorted(children))))
    TP = len(gt_divs & pred_divs)
    FP = len(pred_divs - gt_divs)
    FN = len(gt_divs - pred_divs)
    j = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    return {"TP": TP, "FP": FP, "FN": FN, "jaccard": j}
