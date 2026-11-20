"""Resolve loop roles from the physical topology used to generate MSI."""
from copy import deepcopy

from experiment.doe_campaign import ROLES
from experiment.system_topology import PUMPS
from hardware_devices.capabilities import is_monitor_only


MATERIAL_ROLES = {"": "Unspecified", "reactor": "Model reactor", "AS": "Antisolvent",
                  "solution": "Solution", "solvent": "Solvent", "seed": "Seed suspension",
                  "clean_sol": "Cleaning solvent", "waste": "Waste / discharge"}
QUANTITIES = ("", "Temperature", "Concentration", "FBRM", "D10", "D50", "D90", "AS scale")
PREPARATION_STIRRERS = {"stirring_rate_seed": "seed", "stirring_rate_sol": "solution"}


def preparation_vessels(units, configs, roles, lane_roles, reactor):
    """Find seed and solution vessels from material labels or directed supply assignments."""
    routed = {material: set() for material in PREPARATION_STIRRERS.values()}
    targets = {}
    pump_roles = {device: lane_roles.get(role, {}).get("role", role) for role, device in roles.items()}
    for device, config in configs.items():
        placement = config.get("connectivity", {})
        source = placement.get("source")
        if (config.get("type") not in PUMPS or placement.get("role") != "controlled"
                or source not in units):
            continue
        targets.setdefault(source, set()).add(placement.get("target"))
        material = pump_roles.get(device) or placement.get("experiment_role")
        if material in routed and placement.get("target") == reactor:
            routed[material].add(source)
    return {material: sources or {key for key, unit in units.items()
                                  if unit.get("material_role") == material and (key not in targets or reactor in targets[key])}
            for material, sources in routed.items()}


def resolve_loop_setup(setup, topology, configs, *, campaign=None, applied=None, fallback=None, strict=False):
    """Explicit choices win; infer only unique, compatible, correctly directed links."""
    effective = deepcopy(setup)
    units = {row["code"]: row for row in topology.get("unit_operations", [])}
    used = {p.device_id for exp in campaign.experiments for p in exp.profiles} if campaign else set()
    recording_devices = used | set(campaign.monitor_device_ids) if campaign else configs.keys()
    roles = dict(applied or effective["manual_assignments"])
    origins = {key: "Applied DoE assignments" if applied else "Control Panel assignments" for key in roles}
    messages = []
    for role, device in (fallback or {}).items():
        if role not in roles and role in ROLES and device in used and device not in roles.values():
            roles[role] = device
            origins[role] = "Reused DoE assignment"
    reactor = effective.get("reactor_unit", "")
    if reactor and reactor not in units:
        messages.append(f"The selected model reactor {reactor} no longer exists. Select its current unit in Devices.")
    if not reactor:
        candidates = {configs.get(device, {}).get("connectivity", {}).get("unit")
                      for device in ([roles["T"]] if "T" in roles else used)
                      if configs.get(device, {}).get("type") == "huber"}
        candidates.discard(None)
        candidates.discard("")
        if len(candidates) != 1:
            candidates = {key for key, row in units.items() if row.get("material_role") == "reactor"}
        if len(candidates) == 1:
            reactor = next(iter(candidates))
        elif len(candidates) > 1:
            messages.append("Select the model reactor in Automatic loop setup → Devices; several vessels are available.")
    lane_roles = campaign.line_roles if campaign else {}
    preparation = preparation_vessels(units, configs, roles, lane_roles, reactor)

    def valid_attachment(role, kind, unit):
        if unit == reactor:
            return True
        material = PREPARATION_STIRRERS.get(role)
        return kind == "jkem" and material is not None and unit in preparation[material]

    inferred = {}
    channels = {}
    for device, cfg in configs.items():
        p = cfg.get("connectivity", {})
        if p.get("role", "excluded") == "excluded" or not reactor:
            continue
        kind = cfg.get("type")
        role = p.get("experiment_role", "")
        if kind in PUMPS:
            connected = p.get("target") == reactor
            if connected and not role:
                role = units.get(p.get("source"), {}).get("material_role", "")
        else:
            unit = p.get("unit")
            if not role and not is_monitor_only(cfg):
                if unit == reactor:
                    role = {"huber": "T", "jkem": "stirring_rate", "nidaq_valve": "valve_state"}.get(kind, "")
                elif kind == "jkem":
                    candidates = [aux for aux, material in PREPARATION_STIRRERS.items() if unit in preparation[material]]
                    if len(candidates) == 1:
                        role = candidates[0]
            connected = valid_attachment(role, kind, unit)
        if (connected and role in ROLES and not is_monitor_only(cfg)
                and kind in ROLES[role].compatible_types and (not used or device in used)):
            inferred.setdefault(role, []).append(device)
        quantity = p.get("quantity", "")
        if device in recording_devices and quantity and (p.get("unit") == reactor or
                         (quantity == "AS scale" and units.get(p.get("unit"), {}).get("material_role") == "AS")):
            field = "weight_g" if kind == "mettler" else "temp_external" if kind == "huber" else "value" if kind == "opcua" else None
            if field:
                channels.setdefault(quantity, []).append(f"{device}.{field}")
    for role, candidates in inferred.items():
        if role in roles:
            continue
        candidates = [device for device in candidates if device not in roles.values()]
        if len(candidates) == 1:
            roles[role] = candidates[0]
            origins[role] = "MSI topology"
        elif candidates:
            messages.append(f"Choose {ROLES[role].label}: MSI topology has several candidates ({', '.join(candidates)}).")
    for role, device in roles.items():
        cfg = configs.get(device, {})
        p = cfg.get("connectivity", {})
        if reactor and p.get("role", "excluded") != "excluded":
            kind = cfg.get("type")
            semantic = lane_roles.get(role, {}).get("role", role)
            if kind in PUMPS and p.get("target") != reactor:
                messages.append(f"{device} ({role}) is not directed to / attached to reactor {reactor}. "
                                "Check the MSI source → pump → target direction and the selected reactor.")
            elif kind not in PUMPS and not valid_attachment(semantic, kind, p.get("unit")):
                material = PREPARATION_STIRRERS.get(semantic)
                if material:
                    messages.append(f"{device} must be attached to reactor {reactor} or its {material} preparation vessel. "
                                    "Check the stirrer's MSI unit and the supply pump connection.")
                else:
                    messages.append(f"{device} ({role}) must be attached to reactor {reactor}. Check its MSI unit.")
    for role, options in channels.items():
        if role == "AS scale" and effective["recording"]["no_feed"]:
            continue
        if role not in effective["channels"]:
            if len(options) == 1:
                effective["channels"][role] = options[0]
            else:
                messages.append(f"Choose the {role} recording channel; several probes measure it in the MSI topology.")
    if len(set(roles.values())) != len(roles):
        messages.append("Each loop role needs a distinct device; check the explicit assignments.")
    effective["manual_assignments"] = {role: device for role, device in roles.items() if role in ROLES}
    if strict and messages:
        raise ValueError(" ".join(messages))
    return effective, {"reactor": reactor, "origins": origins, "messages": messages,
                       "topology_channels": {key: values[0] for key, values in channels.items() if len(values) == 1}}
