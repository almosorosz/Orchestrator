"""Read-only values for the selected automatic LOOP starting batch."""
from numbers import Real

from bo.profiles import CONSTRAINTS, TARGETS
from experiment.adaptive_loop import recording_mapping
from experiment.doe_campaign import ROLES
from experiment.loop_process import LOOP_WEIGHT_DEFAULTS, PROCESS_DEFAULTS, VALIDATION_PE_SETTINGS, starting_process_defaults
from imed_python.campaign_dialog import PROCESS_FIELDS
from imed_python.campaign_import import CHANNELS
from imed_python.demo_campaign import model_inputs
from imed_python.pe_settings import PE_WEIGHTS_KEY, PYFIT_DEFAULTS, PYFIT_SETTINGS_KEY, effective_pe_pbm
from imed_python.session import display_name


def value_text(value):
    if value is None:
        return "Not set"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, Real):
        return f"{value:.8g}"
    if isinstance(value, (list, tuple)):
        return " / ".join(value_text(item) for item in value) or "Not set"
    return str(value)


def used_loop_values(setup, backend, context, design_labels):
    campaign, roles, case = (context[key] for key in ("campaign", "roles", "case"))
    experiments = campaign.experiments if campaign else ()
    simulation = experiments[0].imed_simulation if experiments else None
    recording = setup["recording"]
    label = context["label"]
    source = "Automatic loop setup"
    groups = {"Run setup": [
        ("Starting campaign", f"{campaign.name} · {len(experiments)} experiments" if campaign else None, label),
        ("Follow-up experiments", setup["max_followups"], source),
        ("Follow-up crystallization start (s)", 0 if simulation else setup["followup_start_s"],
         "PBM simulation" if simulation else source),
        ("Experiment folder", context["folders"][0] or None, "Settings"),
        ("MODEL folder", context["folders"][1] or None, "Settings"),
    ]}
    if campaign and campaign.doe_factors is not None:
        groups["Run setup"].append(("Sampled DoE factors", ", ".join(campaign.doe_factors) or "None", "Settings DoE"))

    configs, connected, monitors = (context[key] for key in ("configs", "connected", "monitors"))
    assignments = groups["Device assignments"] = []
    for role, device in [*roles.items(), *(("Monitor", device) for device in monitors)]:
        config = configs.get(device, {})
        status = "connected" if device in connected else "disconnected" if config else "missing"
        caption = ROLES[role].label if role in ROLES else role
        name = config.get("name", device)
        assignments.append((caption, f"{device} · {name} · {status}", label))
        if calibration := config.get("pump_calibration"):
            assignments.append((f"{caption} calibration", calibration.get("source_name", "Loaded"), "Control Panel"))
    if not assignments:
        assignments.append(("Assigned devices", None, label))

    profiles = groups["Starting profiles"] = []
    profile_sets = {}
    for experiment in experiments:
        for profile in experiment.profiles:
            profile_sets.setdefault(profile.device_id, []).append(profile)
    for device, plans in profile_sets.items():
        variants = set(plans)
        low = min(p.target for profile in variants for p in profile.points)
        high = max(p.target for profile in variants for p in profile.points)
        unit = next(iter(variants)).unit
        span = f"{low:g}" if low == high else f"{low:g}–{high:g}"
        timing = "Shared profile" if len(variants) == 1 else f"{len(variants)} distinct profiles"
        if len(plans) < len(experiments):
            timing += f" in {len(plans)} of {len(experiments)} runs"
        profiles.append((device, f"{span} {unit} · {timing}", "Starting batch commands"))

    inputs = groups["Starting process inputs"] = []
    saved = next(iter(case.get("ProcInf", [])), {})
    process_labels = {**PROCESS_FIELDS, "InitialConcentration": "Initial concentration (g/g solution)"}
    if simulation:
        _, exp, values, _, mesh = model_inputs(simulation["seeded"])
        values = {**values, "InitialConcentration": exp["ConcentrationData"][1][0]}
        mixed_seed = any(exp.imed_simulation["seeded"] != simulation["seeded"] for exp in experiments)
        if mixed_seed:
            values["SeedMass"] = f"0 / {model_inputs(True)[2]['SeedMass']:g} · depends on starting profile"
    else:
        values = starting_process_defaults(setup, case)
    for key, title in process_labels.items():
        if simulation:
            origin = "PBM simulation"
        elif key in setup["process"]:
            origin = "Process inputs override"
        elif key in saved:
            origin = "First IMED process record"
        elif key in PROCESS_DEFAULTS:
            origin = "Loop default"
        elif key == "InitialConcentration":
            values[key] = recording["initial_concentration"]
            origin = "Recording override" if values[key] is not None else "Measured concentration"
        else:
            origin = "Process inputs"
        inputs.append((title, values.get(key), origin))
    if not simulation:
        for name, overrides in setup["initial_processes"].items():
            for key, value in overrides.items():
                inputs.append((f"{name} · {process_labels.get(key, key)}", value, "Starting experiment override"))

    recorded = groups["Recording"] = []
    if simulation:
        windows = sorted({(exp.imed_simulation["start_s"], exp.imed_simulation["end_s"]) for exp in experiments})
        recorded.append(("Crystallization windows (s)", ", ".join(f"{start:g}–{end:g}" for start, end in windows), "PBM simulation"))
        mapping = {**simulation["monitors"], "Temperature": f"{roles.get('T', '')}.temp_external"}
    else:
        recorded.extend((
            ("Crystallization start (s)", recording["start_s"], "Recording tab"),
            ("Crystallization end (s)", "Last recorded sample" if recording["end_s"] is None else recording["end_s"], "Recording tab"),
            ("Concentration multiplier", recording["concentration_multiplier"], "Recording tab"),
            ("No antisolvent feed", recording["no_feed"], "Recording tab"),
            ("Antisolvent scale direction", "Decreasing" if recording["scale_direction"] == -1 else "Increasing", "Recording tab"),
            ("Scale median window (samples)", recording["smoothing"], "Recording tab"),
        ))
        available = {device: configs[device] for device in {*roles.values(), *monitors} if device in configs}
        try:
            mapping = recording_mapping(setup, roles, available) if "T" in roles else setup["channels"]
        except ValueError as error:
            recorded.append(("Check channel mappings", str(error), "Recording tab"))
            mapping = setup["channels"]
    for role in CHANNELS:
        inherited = "PBM simulation" if simulation else "Recording override" if role in setup["channels"] else "Automatic mapping"
        recorded.append((role, mapping.get(role) or "Not recorded", inherited))
    if not simulation:
        recorded.append(("Seed D10 / D50 / D90 (µm)", recording["seed_percentiles"], "Process inputs"))

    support = groups["Follow-up support"] = []
    for role, profile in setup["support_profiles"].items():
        times, targets = profile["points"]
        support.append((ROLES[role].label, f"{len(times)} points · {min(targets):g}–{max(targets):g} {ROLES[role].unit}",
                        "Support profile override"))
    if pump := setup["seed_pump"]:
        support.append(("Calibrated seed delivery", f"{pump['rpm']:g} RPM · {pump['grams_per_second']:g} g/s", "Devices"))

    model = groups["Modeling"] = []
    if backend == "bo" and (bo := context["bo"]) is not None:
        model.append(("Selected prior observations", len(bo["selected"]), "BO page"))
        for key, target in bo["targets"].items():
            if target["enabled"]:
                mode = f"Match {target['target']:g} {TARGETS[key][1]}" if target["mode"] == "match" else target["mode"]
                model.append((TARGETS[key][0], f"{mode} · weight {target['weight']:g}", "BO page"))
        for key, bounds in bo["constraints"].items():
            if any(bound is not None for bound in bounds):
                name, unit = CONSTRAINTS[key]
                model.append((f"{name} bounds ({unit})", bounds, "BO page"))
        for key, bounds in bo["output_bounds"].items():
            if any(bound is not None for bound in bounds):
                name, unit, _ = TARGETS[key]
                model.append((f"{name} output bounds ({unit})", bounds, "BO page"))
    elif backend == "imed":
        stored = case.get(PE_WEIGHTS_KEY, case.get("objweightfactor", {}))
        stored = stored if isinstance(stored, dict) else {}
        options = setup["imed"]
        pbm_case = {**case, "pbmsetting": {**mesh, **case.get("pbmsetting", {})}} if simulation else case
        pbm = effective_pe_pbm(pbm_case)
        candidates = [display_name(item, f"Model {i}") for i, item in enumerate(case.get("ModelParameter", []), 1)]
        fit_defaults = VALIDATION_PE_SETTINGS if simulation and not candidates else {}
        fit = {**PYFIT_DEFAULTS, **case.get(PYFIT_SETTINGS_KEY, fit_defaults)}
        model.extend((
            ("Candidate models", ", ".join(candidates) or ("Validation PBM" if simulation else None), "IMED case"),
            ("PE solver / mesh bins", f"{pbm.get('Numerics', 'conservative')} / {pbm['N']}", "IMED case"),
            ("PE weights · count / concentration / size",
             [stored.get(f"W{i}", default) for i, default in enumerate(LOOP_WEIGHT_DEFAULTS, 1)], "IMED case"),
            ("PE size samples", context["size_term"], "Settings"),
            ("CPU workers", context["workers"], "IMED case"),
            ("PE global iterations / population", [fit["global_iterations"], fit["population_size"]], "PE settings"),
            ("PE local evaluations", fit["local_evaluations"], "PE settings"),
            ("Model survival threshold (ΔAIC)", options["delta_aic"], "IMED design"),
            ("Design search iterations", options["iterations"], "IMED design"),
            ("Covariance samples", options["covariance_samples"], "IMED design"),
            ("Final equilibration (s)", options["equilibration_s"], "IMED design"),
        ))
        for key, title in design_labels.items():
            model.append((title, options["conditions"].get(key), "IMED design"))
    return groups
