"""Save edited IMED parameters and settings while preserving other MAT variables."""
from io import BytesIO
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.io import loadmat, savemat
from scipy.io.matlab import varmats_from_mat


def save_model(source, destination, edits, settings=None, *, preserve_legacy=False,
               allow_overwrite=False):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() and not allow_overwrite:
        raise ValueError("Choose a new filename to preserve the original IMED model")
    original = source.read_bytes()
    endian = {b"IM": "little", b"MI": "big"}.get(original[126:128])
    if endian is None:
        raise ValueError("Saving requires an IMED MATLAB v5-format model")

    updates = dict(settings or {})
    if not preserve_legacy:
        updates.pop("gasetting", None)
    # SciPy emits little-endian v5 matrices. Mixing byte orders would corrupt the file.
    if (updates or edits) and endian != "little":
        raise ValueError("Saving edits requires a little-endian IMED model")
    if edits:
        models = loadmat(BytesIO(original), variable_names=["ModelParameter"])["ModelParameter"]
        for index, values in edits.items():
            model = models[np.unravel_index(index, models.shape, order="F")]
            model["paraset"] = values["paraset"]
            for field in ("para", "covmat", "resnorm", "resid", "jacobian", "nvars", "SSR",
                          "r", "FVector", "AICVector", "t_test", "Results"):
                if field in models.dtype.names:
                    model[field] = values.get(field, np.empty((0, 0)))
        updates["ModelParameter"] = models

    replacements = {}
    if updates:
        replacement = BytesIO()
        savemat(replacement, updates, do_compression=True)
        replacement.seek(0)
        replacements = {name: stream.getvalue()[128:]
                        for name, stream in varmats_from_mat(replacement)}
    parts = []
    offset = 128
    shift = 0
    subsystem_offset = int.from_bytes(original[116:124], endian)
    seen = set()
    for name, stream in varmats_from_mat(BytesIO(original)):
        part = stream.getvalue()[128:]
        new_part = part
        if name == "gasetting" and not preserve_legacy:
            # New Python copies omit the legacy MATLAB optimizer options.
            new_part = b""
        elif name in replacements:
            if name in seen:
                raise ValueError(f"Unable to safely replace duplicate MAT variable {name}")
            seen.add(name)
            new_part = replacements[name]
        if subsystem_offset > offset:
            shift += len(new_part) - len(part)
        parts.append(new_part)
        offset += len(part)
    if offset != len(original) or (edits and "ModelParameter" not in seen):
        raise ValueError("Unable to safely replace the MAT records")
    parts.extend(part for name, part in replacements.items() if name not in seen)
    header = bytearray(original[:128])
    if subsystem_offset:
        header[116:124] = (subsystem_offset + shift).to_bytes(8, endian)
    payload = b"".join([header, *parts])
    _atomic_write(destination, payload)


def _atomic_write(destination, payload):
    temporary = None
    try:
        with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


CHECKPOINT_FIELDS = (
    "ExpData", "ProcInf", "Solubility", "ModelParameter", "pbmsetting",
    "objweightfactor", "pyfitsettings", "pepbmsettings", "peweights", "PythonWorkflow",
)
MATLAB_FIT_SUMMARIES = ("MDIteration", "MDsimulation", "PPsimulation", "summaryTable")


def checkpoint_variables(case):
    """Convert current IMED case fields to MATLAB-compatible records."""
    updates = {key: case[key] for key in CHECKPOINT_FIELDS if key in case}
    for key in MATLAB_FIT_SUMMARIES:
        value = case.get(key)
        if isinstance(value, np.ndarray) and not value.size:
            updates[key] = value
    # Write empty records when an editor explicitly removes the last experiment.
    # Empty solver inputs are normalized on read.
    if isinstance(updates.get("Solubility"), dict) and not updates["Solubility"]:
        updates["Solubility"] = np.empty((0, 0))
    # Keep MATLAB's struct arrays for the existing IMED input and model records.
    for key in ("ExpData", "ProcInf", "ModelParameter"):
        records = updates.get(key)
        if isinstance(records, (list, tuple)) and records:
            fields = list(dict.fromkeys(field for record in records for field in record))
            structs = np.empty((1, len(records)), dtype=[(field, object) for field in fields])
            for index, record in enumerate(records):
                for field in fields:
                    structs[0, index][field] = record.get(field, np.empty((0, 0)))
            updates[key] = structs
    return updates


def save_checkpoint(source, destination, case):
    """Write the current case and workflow while preserving opaque source variables."""
    destination = Path(destination)
    updates = checkpoint_variables(case)
    if source is not None:
        save_model(source, destination, {}, updates, preserve_legacy=True, allow_overwrite=True)
    else:
        stream = BytesIO()
        savemat(stream, updates, do_compression=True)
        _atomic_write(destination, stream.getvalue())
