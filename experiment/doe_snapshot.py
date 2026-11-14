"""JSON snapshots of exact sampled conditions and generated profiles; never re-sample on load."""
from copy import deepcopy
import numpy as np
import pandas as pd

STORAGE = ("data", "data_omni", "profile", "profile_omni", "DoE")
SCALARS = ("TYPE", "dimensions", "samples", "Fract", "if_sampling", "actually_sampled_experiments")
ARRAYS = ("DoE_samples", "DoE_sampled_scaled")


def snapshot_doe(creator):
    if creator is None:
        return None
    return {"version": 1,
        "storage": {key: {name: frame.astype(object).where(pd.notna(frame), None).to_dict(orient="split")
                          for name, frame in getattr(creator.exp_storage, key, {}).items()} for key in STORAGE},
        "attributes": {key: getattr(creator, key) for key in SCALARS},
        "arrays": {key: np.asarray(getattr(creator, key)).tolist() for key in ARRAYS},
        "DoE_val": deepcopy(creator.DoE_val), "weights": list(creator.weights),
        "parameter_snapshots": deepcopy(creator.parameter_snapshots)}


def validate_snapshot(snapshot):
    if snapshot is None:
        return
    if snapshot.get("version") != 1:
        raise ValueError("Unsupported generated DoE snapshot version")
    data = snapshot["storage"]["data"]
    if not data or snapshot["attributes"]["actually_sampled_experiments"] != len(data):
        raise ValueError("Generated DoE snapshot has inconsistent experiment counts")
    for name, saved in data.items():
        frame = pd.DataFrame(**saved)
        if frame.empty or "t" not in frame or not frame.columns.is_unique:
            raise ValueError(f"Invalid generated profile: {name}")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (frame["t"] < 0).any() or (frame["t"].diff().dropna() < 0).any():
            raise ValueError(f"Invalid generated profile values/timeline: {name}")
    for values in snapshot.get("arrays", {}).values():
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError("Sampled DoE values must be finite")


def restore_doe(service, snapshot):
    if snapshot is None:
        return None
    validate_snapshot(snapshot)
    creator = service.build_creator()
    for key in STORAGE:
        setattr(creator.exp_storage, key, {name: pd.DataFrame(**saved)
                for name, saved in snapshot["storage"].get(key, {}).items()})
    for key in SCALARS:
        setattr(creator, key, snapshot["attributes"][key])
    for key in ARRAYS:
        setattr(creator, key, np.asarray(snapshot["arrays"][key], dtype=float))
    creator.DoE_val = deepcopy(snapshot["DoE_val"])
    creator.weights = list(snapshot["weights"])
    creator.parameter_snapshots = deepcopy(snapshot["parameter_snapshots"])
    return creator
