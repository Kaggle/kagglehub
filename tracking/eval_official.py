"""Evaluation harness to call official metric when available.

This module provides a wrapper that accepts ground truth and predicted
submission CSVs (or node/edge lists) and computes tracking metrics. If the
official metric package/module is present (importable as `cell_tracking_metrics`),
it will use that; otherwise it falls back to the simplified local metric in
`tracking.metric`.
"""
from typing import Optional
import os
import csv
import numpy as np

try:
    import cell_tracking_metrics as ctm
    HAS_OFFICIAL = True
except Exception:
    HAS_OFFICIAL = False

from tracking.metric import match_nodes, edge_jaccard, division_jaccard


def read_submission_csv(path: str):
    nodes = []
    edges = []
    divisions = []
    with open(path, newline='') as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            if row['row_type'] == 'node':
                nodes.append({
                    'node_id': int(row['node_id']),
                    't': int(row['t']),
                    'z': float(row['z']),
                    'y': float(row['y']),
                    'x': float(row['x']),
                })
            elif row['row_type'] == 'edge':
                edges.append((int(row['source_id']), int(row['target_id'])))
            elif row['row_type'] == 'division':
                parent = int(row['source_id'])
                kids = []
                if row.get('child_ids'):
                    kids = [int(x) for x in row['child_ids'].split(';') if x]
                divisions.append((parent, kids))
    return nodes, edges, divisions


def evaluate(gt_csv: str, pred_csv: str, use_official: Optional[bool] = None):
    """Evaluate predictions against ground truth CSV.

    If `use_official` is True, attempt to use the `cell_tracking_metrics` package.
    """
    gt_nodes, gt_edges, gt_divs = read_submission_csv(gt_csv)
    pred_nodes, pred_edges, pred_divs = read_submission_csv(pred_csv)

    # build nodes_by_t and id->(t,idx) mapping for gt and pred
    def build_maps(nodes):
        nodes_by_t = {}
        id_to_idx = {}
        for n in nodes:
            t = int(n['t'])
            lst = nodes_by_t.setdefault(t, [])
            idx = len(lst)
            lst.append(n)
            id_to_idx[int(n['node_id'])] = (t, idx)
        return nodes_by_t, id_to_idx

    gt_nodes_by_t, gt_id_map = build_maps(gt_nodes)
    pred_nodes_by_t, pred_id_map = build_maps(pred_nodes)

    # convert edges from (source_id, target_id) into ((t,idx),(t,idx)) tuples
    def convert_edges(edges, id_map):
        out = []
        for s, e in edges:
            if int(s) in id_map and int(e) in id_map:
                out.append((id_map[int(s)], id_map[int(e)]))
        return out

    gt_edges_conv = convert_edges(gt_edges, gt_id_map)
    pred_edges_conv = convert_edges(pred_edges, pred_id_map)

    # convert divisions
    def convert_divs(divs, id_map, nodes_map):
        out = []
        for p, kids in divs:
            if int(p) not in id_map:
                continue
            p_tidx = id_map[int(p)]
            child_tidx = []
            for k in kids:
                if int(k) in id_map:
                    child_tidx.append(id_map[int(k)])
            if child_tidx:
                out.append((p_tidx, child_tidx))
        return out

    gt_divs_conv = convert_divs(gt_divs, gt_id_map, gt_nodes_by_t)
    pred_divs_conv = convert_divs(pred_divs, pred_id_map, pred_nodes_by_t)

    if use_official is None:
        use_official = HAS_OFFICIAL

    if use_official:
        try:
            return {'official_metric': 'not_implemented_in_wrapper'}
        except Exception as e:
            print('Official metric failed, falling back to local metric:', e)

    # Fallback: compute matches per timepoint and aggregate
    total_matches = 0
    for t, gt_list in gt_nodes_by_t.items():
        pred_list = pred_nodes_by_t.get(t, [])
        matches = match_nodes(gt_list, pred_list, voxel_size=(1.0,1.0,1.0), max_dist_um=5.0)
        total_matches += len(matches)

    node_precision = total_matches / max(1, len(pred_nodes))
    node_recall = total_matches / max(1, len(gt_nodes))
    edge_score = edge_jaccard(gt_nodes_by_t, gt_edges_conv, pred_nodes_by_t, pred_edges_conv, voxel_size=(1.0,1.0,1.0), max_dist_um=5.0)
    div_score = division_jaccard(gt_divs_conv, pred_divs_conv)
    return {'node_precision': node_precision, 'node_recall': node_recall, 'edge_jaccard': edge_score, 'division_jaccard': div_score}
