Tracking harness
===============

This folder contains a minimal evaluation harness for the 3D cell-tracking competition.

Quick start
-----------

1. Install dependencies (recommended in a virtualenv):

```bash
pip install numpy zarr scipy pandas
```

2. Use the loader and metric

Python example:

```python
from tracking.data_loader import open_zarr, iter_frames, centroids_from_labels
from tracking.metric import match_nodes, edge_jaccard

# open a zarr file
arr, voxel = open_zarr('path/to/volume.zarr')

# iterate frames
for t, frame in iter_frames(arr['images'] if 'images' in arr else arr):
    print('frame', t, frame.shape)

# load label zarr and extract centroids
# labels_arr can be loaded via zarr.open('path/to/labels.zarr')[:]
# centroids = centroids_from_labels(labels_arr, voxel_size=(1.625, 0.40625, 0.40625))

```

Notes
-----
- These are intentionally small helper functions to get started quickly. For production use, add robust I/O,
  tiling, memory-mapped access, and parallelization.
- The metric implementation is simplified and intended for local experiments; for exact competition scoring use the
  official metric reference in the competition repo.
