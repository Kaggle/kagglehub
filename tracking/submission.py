"""Helpers to construct and write submission CSV for the competition.

The submission format requires rows for nodes and edges with exactly these columns:
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id

Provide `write_submission_csv(dataset_name, nodes, edges, out_path)` where nodes is a list
of dicts with keys `node_id,t,z,y,x` and edges is list of dicts with `source_id,target_id` where ids refer to node_id.
"""

import csv
from typing import List, Dict


def _validate_rows(nodes: List[Dict], edges: List[Dict]):
    node_ids = set()
    for node in nodes:
        node_id = int(node.get("node_id", -1))
        if node_id < 0 or node_id in node_ids:
            raise ValueError("Node IDs must be unique non-negative integers")
        node_ids.add(node_id)
        for key in ("t", "z", "y", "x"):
            if key not in node:
                raise ValueError(f"Node is missing required field: {key}")
    for edge in edges:
        source_id = int(edge.get("source_id", -1))
        target_id = int(edge.get("target_id", -1))
        if source_id not in node_ids or target_id not in node_ids:
            raise ValueError("Every edge must reference nodes in the same dataset")


def write_submission_csv(
    dataset: str, nodes: List[Dict], edges: List[Dict], out_path: str, divisions: List[Dict] = None
):
    """Write nodes and edges, encoding divisions as additional edges.

    Divisions should be a list of dicts with keys `parent_id` and `child_ids` (list).
    The competition identifies divisions from nodes with multiple outgoing edges,
    so no separate division row type is written.
    """
    if divisions is None:
        divisions = []
    fieldnames = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    all_edges = list(edges)
    for division in divisions:
        parent_id = division.get("parent_id", -1)
        for child_id in division.get("child_ids", []):
            edge = {"source_id": parent_id, "target_id": child_id}
            if edge not in all_edges:
                all_edges.append(edge)
    _validate_rows(nodes, all_edges)
    idx = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # write nodes
        for n in nodes:
            row = {
                "id": idx,
                "dataset": dataset,
                "row_type": "node",
                "node_id": int(n.get("node_id", -1)),
                "t": int(n.get("t", -1)),
                "z": int(n.get("z", -1)),
                "y": int(n.get("y", -1)),
                "x": int(n.get("x", -1)),
                "source_id": -1,
                "target_id": -1,
            }
            writer.writerow(row)
            idx += 1
        # write edges
        for e in all_edges:
            row = {
                "id": idx,
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": e.get("source_id", -1),
                "target_id": e.get("target_id", -1),
            }
            writer.writerow(row)
            idx += 1
