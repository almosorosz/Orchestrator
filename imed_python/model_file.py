"""Save edited IMED parameters and settings while preserving other MAT variables."""
from io import BytesIO
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.io import loadmat, savemat
from scipy.io.matlab import varmats_from_mat


def save_model(source, destination, edits, settings=None):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Choose a new filename to preserve the original IMED model")
    original = source.read_bytes()
    endian = {b"IM": "little", b"MI": "big"}.get(original[126:128])
    if endian is None:
        raise ValueError("Saving requires an IMED MATLAB v5-format model")

    updates = dict(settings or {})
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
        if name == "gasetting":
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

    temporary = None
    try:
        with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
