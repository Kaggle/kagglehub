"""Simple pipeline to extract nodes from label volumes and write a submission CSV.

Usage:
  python3 -m tracking.pipeline --dataset 44b6 --labels path/to/labels.zarr --out submission.csv

Supports: .zarr (uses zarr), .npz/.npy (uses numpy load).
"""

import os
import argparse
import glob
import csv
import numpy as np


def load_labels(path):
    if path.endswith(".zarr") or os.path.isdir(path):
        try:
            import zarr
        except Exception:
            raise RuntimeError("zarr not installed; install with `pip install zarr`")
        arr = zarr.open(path, mode="r")
        # prefer a dataset named 'labels' or 'masks'
        if "labels" in arr:
            data = np.asarray(arr["labels"])
        elif "masks" in arr:
            data = np.asarray(arr["masks"])
        else:
            # take the root array
            data = np.asarray(arr)
        return data
    elif path.endswith(".npz") or path.endswith(".npy"):
        data = np.load(path)
        # if npz, get first array
        if isinstance(data, np.lib.npyio.NpzFile):
            keys = list(data.keys())
            if not keys:
                raise RuntimeError("Empty npz file")
            arr = data[keys[0]]
        else:
            arr = data
        return arr
    else:
        raise RuntimeError("Unsupported labels path; provide .zarr or .npz/.npy")


def run(
    dataset: str,
    labels_path: str,
    out_csv: str,
    tile_size=(64, 64, 64),
    overlap=(16, 16, 16),
    merge_distance=3.0,
    link=False,
    max_dist_um=7.0,
    voxel_size=(1.625, 0.40625, 0.40625),
    images_path=None,
    appearance_weight=0.0,
    use_kalman=False,
):
    from tracking.postprocess import extract_nodes_from_label_tiles
    from tracking.submission import write_submission_csv
    from tracking.linking import link_nodes, link_nodes_with_appearance
    from tracking.kalman import link_nodes_with_kalman
    from tracking.division import detect_divisions_enhanced

    labels = load_labels(labels_path)
    nodes = extract_nodes_from_label_tiles(
        labels,
        tile_size=tuple(map(int, tile_size)),
        overlap=tuple(map(int, overlap)),
        merge_distance_voxels=float(merge_distance),
    )
    edges = []
    if link:
        if use_kalman:
            # load images if provided
            images = None
            if images_path:
                if images_path.endswith(".npz") or images_path.endswith(".npy"):
                    import numpy as _np

                    data = _np.load(images_path)
                    if isinstance(data, _np.lib.npyio.NpzFile):
                        images = data[list(data.keys())[0]]
                    else:
                        images = data
                else:
                    try:
                        import zarr as _z
                    except Exception:
                        raise RuntimeError("zarr is required to load image zarr stores")
                    arr = _z.open(images_path, mode="r")
                    images = _np.asarray(arr)
            edges = link_nodes_with_kalman(
                nodes,
                images=images,
                max_dist_um=float(max_dist_um),
                voxel_size=tuple(map(float, voxel_size)),
                appearance_weight=float(appearance_weight),
            )
        elif images_path and float(appearance_weight) > 0.0:
            # load images
            if images_path.endswith(".npz") or images_path.endswith(".npy"):
                import numpy as _np

                data = _np.load(images_path)
                if isinstance(data, _np.lib.npyio.NpzFile):
                    images = data[list(data.keys())[0]]
                else:
                    images = data
            else:
                try:
                    import zarr as _z
                except Exception:
                    raise RuntimeError("zarr is required to load image zarr stores")
                arr = _z.open(images_path, mode="r")
                images = _np.asarray(arr)
            edges = link_nodes_with_appearance(
                nodes,
                images,
                max_dist_um=float(max_dist_um),
                voxel_size=tuple(map(float, voxel_size)),
                appearance_weight=float(appearance_weight),
            )
        else:
            edges = link_nodes(nodes, max_dist_um=float(max_dist_um), voxel_size=tuple(map(float, voxel_size)))
    # attempt enhanced division detection using the raw label volume and optional images
    divisions_out = []
    try:
        images_arr = None
        # labels already loaded as `labels`
        if images_path:
            if images_path.endswith(".npz") or images_path.endswith(".npy"):
                data = np.load(images_path)
                if isinstance(data, np.lib.npyio.NpzFile):
                    images_arr = data[list(data.keys())[0]]
                else:
                    images_arr = data
            else:
                try:
                    import zarr as _z
                except Exception:
                    _z = None
                if _z is None:
                    images_arr = None
                else:
                    arr = _z.open(images_path, mode="r")
                    images_arr = np.asarray(arr)
        divs = detect_divisions_enhanced(nodes, labels, images=images_arr, voxel_size=tuple(map(float, voxel_size)))
        for p, kids, score in divs:
            divisions_out.append(
                {"parent_id": int(p["node_id"]), "child_ids": [int(k["node_id"]) for k in kids], "score": float(score)}
            )
    except Exception:
        divisions_out = []

    write_submission_csv(dataset, nodes, edges, out_csv, divisions=divisions_out)
    print(
        f"Wrote submission CSV with {len(nodes)} nodes, {len(edges)} edges and {len(divisions_out)} divisions to {out_csv}"
    )


def run_directory(input_dir: str, out_csv: str, **kwargs):
    """Generate one submission CSV from all label stores in a test directory.

    Dataset names come from directory names for .zarr stores and file stems for
    .npz/.npy fixtures. Use a separate images directory when appearance linking
    is required; matching image stores are selected by dataset name.
    """
    stores = sorted(glob.glob(os.path.join(input_dir, "*.zarr")))
    stores += sorted(glob.glob(os.path.join(input_dir, "*_labels.npz")))
    stores += sorted(glob.glob(os.path.join(input_dir, "*_labels.npy")))
    if not stores:
        raise RuntimeError(f"No .zarr, .npz, or .npy label stores found in {input_dir}")

    from tracking.postprocess import extract_nodes_from_label_tiles
    from tracking.submission import _validate_rows
    from tracking.linking import link_nodes

    all_nodes = []
    all_edges = []
    next_id = 1
    for store in stores:
        dataset = os.path.splitext(os.path.basename(store))[0]
        if dataset.endswith("_labels"):
            dataset = dataset[:-7]
        labels = load_labels(store)
        nodes = extract_nodes_from_label_tiles(
            labels,
            tile_size=tuple(map(int, kwargs.get("tile_size", (64, 64, 64)))),
            overlap=tuple(map(int, kwargs.get("overlap", (16, 16, 16)))),
            merge_distance_voxels=float(kwargs.get("merge_distance", 3.0)),
        )
        for node in nodes:
            node["node_id"] = next_id
            next_id += 1
            all_nodes.append((dataset, node))
        if kwargs.get("link", False):
            local_edges = link_nodes(
                nodes,
                max_dist_um=float(kwargs.get("max_dist_um", 7.0)),
                voxel_size=tuple(map(float, kwargs.get("voxel_size", (1.625, 0.40625, 0.40625)))),
            )
            all_edges.extend((dataset, edge) for edge in local_edges)
        print(f"Prepared {dataset}: {len(nodes)} nodes")

    node_values = [node for _, node in all_nodes]
    _validate_rows(node_values, [edge for _, edge in all_edges])
    fieldnames = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        index = 0
        for dataset, node in all_nodes:
            writer.writerow(
                {
                    "id": index,
                    "dataset": dataset,
                    "row_type": "node",
                    "node_id": int(node["node_id"]),
                    "t": int(node["t"]),
                    "z": int(round(node["z"])),
                    "y": int(round(node["y"])),
                    "x": int(round(node["x"])),
                    "source_id": -1,
                    "target_id": -1,
                }
            )
            index += 1
        for dataset, edge in all_edges:
            writer.writerow(
                {
                    "id": index,
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": edge["source_id"],
                    "target_id": edge["target_id"],
                }
            )
            index += 1
    print(f"Wrote {out_csv} with {len(all_nodes)} nodes and {len(all_edges)} edges")


def _parse_tuple(s):
    return tuple(int(x) for x in s.split(","))


def main():
    p = argparse.ArgumentParser()
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dataset", help="dataset name for a single labels store")
    input_group.add_argument("--input-dir", help="directory containing one labels .zarr per dataset")
    p.add_argument("--labels", help="labels store for a single dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--tile-size", default="64,64,64", help="tile size Z,Y,X")
    p.add_argument("--overlap", default="16,16,16", help="overlap Z,Y,X")
    p.add_argument("--merge-distance", default=3.0, type=float)
    p.add_argument("--link", action="store_true", help="perform frame-to-frame linking to produce edges")
    p.add_argument("--max-dist", default=7.0, type=float, help="max link distance in microns")
    p.add_argument("--voxel-size", default="1.625,0.40625,0.40625", help="voxel size (z,y,x) in microns")
    p.add_argument("--images", default=None, help="optional images zarr/.npz path for appearance-aware linking")
    p.add_argument(
        "--appearance-weight", default=0.0, type=float, help="weight for appearance in linking (0=distance-only)"
    )
    p.add_argument("--use-kalman", action="store_true", help="use Kalman motion prior sequential linking")
    args = p.parse_args()
    vs = tuple(float(x) for x in args.voxel_size.split(","))
    if args.input_dir:
        run_directory(
            args.input_dir,
            args.out,
            tile_size=_parse_tuple(args.tile_size),
            overlap=_parse_tuple(args.overlap),
            merge_distance=args.merge_distance,
            link=args.link,
            max_dist_um=args.max_dist,
            voxel_size=vs,
        )
        return
    if not args.labels:
        p.error("--labels is required with --dataset")
    run(
        args.dataset,
        args.labels,
        args.out,
        tile_size=_parse_tuple(args.tile_size),
        overlap=_parse_tuple(args.overlap),
        merge_distance=args.merge_distance,
        link=args.link,
        max_dist_um=args.max_dist,
        voxel_size=vs,
        images_path=args.images,
        appearance_weight=args.appearance_weight,
        use_kalman=args.use_kalman,
    )


if __name__ == "__main__":
    main()
