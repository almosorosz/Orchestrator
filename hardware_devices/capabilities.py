def is_monitor_only(config):
    """A writable instrument may be used as a passive probe in this setup."""
    return config.get("type") in {"opcua", "mettler"} or config.get("connectivity", {}).get("role") == "read_only"
