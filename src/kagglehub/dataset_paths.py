import ntpath
import posixpath


def normalize_dataset_path(path: str) -> str:
    """Return a canonical, dataset-relative path or raise for unsafe input."""
    if not isinstance(path, str):
        msg = "Dataset path must be a string"
        raise TypeError(msg)
    if "\x00" in path:
        msg = "Dataset path cannot contain a null byte"
        raise ValueError(msg)
    if ntpath.splitdrive(path)[0]:
        msg = f"Dataset path must be relative: {path!r}"
        raise ValueError(msg)

    path = path.replace("\\", "/")
    if path.startswith("/"):
        msg = f"Dataset path must be relative: {path!r}"
        raise ValueError(msg)

    parts = [part for part in path.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        msg = f"Dataset path must stay within the dataset: {path!r}"
        raise ValueError(msg)

    return posixpath.join(*parts)


def validate_dataset_path_descendant(parent: str, candidate: str) -> str:
    """Validate a server-returned path before it is used for a local write."""
    normalized_parent = normalize_dataset_path(parent)
    normalized_candidate = normalize_dataset_path(candidate)
    if not normalized_candidate.startswith(f"{normalized_parent}/"):
        msg = f"Dataset API returned a path outside {normalized_parent!r}: {candidate!r}"
        raise ValueError(msg)
    return normalized_candidate
