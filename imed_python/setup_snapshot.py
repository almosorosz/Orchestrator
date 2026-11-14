"""Embed an IMED checkpoint in the versioned CrystalLoop setup preset."""
from base64 import b64decode
from io import BytesIO

from .native_case import pack_case, unpack_case
from .solver import read_model


def pack_imed_settings(case):
    return {"active_workflow": "imed", "imed_case": pack_case(case)}


def read_imed_settings(settings):
    if settings is None:
        return None
    if not isinstance(settings, dict) or settings.get("active_workflow") not in ("imed", "bo"):
        raise ValueError("Unsupported modeling workflow in the setup preset.")
    if "imed_case" in settings:
        return unpack_case(settings["imed_case"])
    encoded = settings.get("imed_mat")
    if not isinstance(encoded, str):
        raise ValueError("The setup preset is missing its IMED checkpoint.")
    try:
        with BytesIO(b64decode(encoded, validate=True)) as stream:
            case = read_model(stream, allow_settings_only=True)
    except (ValueError, OSError, TypeError, IndexError) as error:
        raise ValueError("The setup preset contains an invalid IMED checkpoint.") from error
    return case
