"""Embed an IMED checkpoint in the versioned CrystalLoop setup preset."""
from base64 import b64decode, b64encode
from io import BytesIO

from scipy.io import savemat

from .model_file import checkpoint_variables
from .solver import read_model


def pack_imed_settings(case):
    payload = BytesIO()
    savemat(payload, checkpoint_variables(case), do_compression=True)
    return {"active_workflow": "imed", "imed_mat": b64encode(payload.getvalue()).decode("ascii")}


def read_imed_settings(settings):
    if settings is None:
        return None
    if not isinstance(settings, dict) or settings.get("active_workflow") != "imed":
        raise ValueError("Unsupported modeling workflow in the setup preset.")
    encoded = settings.get("imed_mat")
    if not isinstance(encoded, str):
        raise ValueError("The setup preset is missing its IMED checkpoint.")
    try:
        with BytesIO(b64decode(encoded, validate=True)) as stream:
            case = read_model(stream, allow_settings_only=True)
    except (ValueError, OSError, TypeError, IndexError) as error:
        raise ValueError("The setup preset contains an invalid IMED checkpoint.") from error
    return case
