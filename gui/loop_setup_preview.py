"""Read-only values for the selected automatic LOOP starting batch."""
from numbers import Real

from bo.profiles import CONSTRAINTS, TARGETS, selected_horizon
from experiment.adaptive_loop import (
    antisolvent_recording_source, imed_objective_channels, imed_response_weights, recording_mapping,
)
from experiment.doe_campaign import ROLES
from experiment.doe_decisions import LABELS, decision_groups, decision_summary, starting_run_process
from experiment.run_stages import modeling_campaign
from experiment.loop_process import PROCESS_DEFAULTS, PROCESS_FIELDS, VALIDATION_PE_SETTINGS, starting_process_defaults
from imed_python.campaign_import import CHANNELS
from imed_python.demo_campaign import legacy_simulation, model_inputs
from imed_python.pe_settings import PYFIT_DEFAULTS, PYFIT_SETTINGS_KEY, effective_pe_pbm
from imed_python.session import display_name


PREVIEW_AMOUNTS = {key: PROCESS_FIELDS[key] for key in ("SolvMass", "AntiSolvMass", "SeedMass")}
PREVIEW_AMOUNTS["InitialVolume_ml"] = "Initial liquid volume (mL)"


def starting_input_rows(setup, context):
    """Resolve draft values with the same starting profiles and settings as execution."""
    setup = context.get("effective_setup", setup)
    campaign = modeling_campaign(context.get("campaign"), setup)
    if not campaign:
        return {}
    case = context["case"]
    defaults = starting_process_defaults(setup, case)
    base_sources = dict.fromkeys(PROCESS_DEFAULTS, "Loop default")
    base_sources.update(dict.fromkeys(next(iter(case.get("ProcInf", [])), {}), "First IMED process record"))
    base_sources.update(dict.fromkeys(setup["process"], "Common process input"))
    concentration = setup["recording"]["initial_concentration"]
    if concentration is not None and "InitialConcentration" not in defaults:
        defaults["InitialConcentration"] = concentration
        base_sources["InitialConcentration"] = "Recording setting"
    rows = {}
    for exp in campaign.experiments:
        sources, issues = base_sources.copy(), {}
        values = starting_run_process(exp, setup, context["roles"], context["configs"], campaign.line_roles,
                                      defaults, sources=sources, issues=issues)
        rows[exp.name] = values, sources, issues, legacy_simulation(exp.imed_simulation) is None
    return rows


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


def uses_followup_offset(period_mode, context):
    campaign = context.get("campaign")
    first = campaign.experiments[0] if campaign and campaign.experiments else None
    return (period_mode == "definition" and not context.get("run_stages")
            and (first is None or not (first.run_stages or first.model_window or legacy_simulation(first.imed_simulation))))


def used_loop_values(setup, backend, context, design_labels):
    setup = context.get("effective_setup", setup)
    campaign, roles, case = (context[key] for key in ("campaign", "roles", "case"))
    experiments = campaign.experiments if campaign else ()
    simulation = legacy_simulation(experiments[0].imed_simulation) if experiments else None
    recording = setup["recording"]
    label = context["label"]
    source = "Automatic loop setup"
    groups = {"Run setup": [
        ("Starting campaign", f"{campaign.name} · {len(experiments)} experiments" if campaign else None, label),
        ("Follow-up experiments", setup["max_followups"], source),
        ("Experiment folder", context["folders"][0] or None, "Settings"),
        ("MODEL folder", context["folders"][1] or None, "Settings"),
    ]}
    if uses_followup_offset(setup["experiment_period"]["mode"], context):
        groups["Run setup"].insert(2, ("Start delay for unstaged profiles (s)", setup["followup_start_s"], source))
    elif experiments and experiments[0].run_stages and not simulation:
        groups["Run setup"].insert(2, ("Follow-up scope", "Selected stages only", "Experiment period"))
    if campaign and campaign.doe_factors is not None:
        groups["Run setup"].append(("Sampled DoE factors", ", ".join(campaign.doe_factors) or "None", "Settings DoE"))
    if (decisions := context.get("doe_decisions")) is not None:
        active = decision_groups(decisions, imed=backend == "imed")
        if experiments and experiments[0].run_stages and not simulation:
            active -= {"seed", "volume"}
        if backend == "imed" and not setup["mixing"]["enabled"]:
            active.discard("stirring")
        groups["DoE decision variables"] = [("Varies", decision_summary(decisions, active), label)]
        for key in (group for group in LABELS if group in active):
            levels = decisions["levels"].get(key) if key != "stirring" or backend == "imed" else None
            detail = (levels[0], levels[-1]) if levels else f"{len(decisions['traces'][key])} profile choices"
            groups["DoE decision variables"].append((LABELS[key], detail, label))
        if backend == "imed" and "stirring" in decision_groups(decisions, imed=True):
            detail = ("Available when a surviving model uses the power term" if setup["mixing"]["enabled"]
                      else "Fixed while average mixing is disabled")
            groups["DoE decision variables"].append(("IMED stirring", detail, "IMED design"))

    configs, connected, monitors = (context[key] for key in ("configs", "connected", "monitors"))
    assignments = groups["Device assignments"] = []
    for role, device in [*roles.items(), *(("Monitor", device) for device in monitors)]:
        config = configs.get(device, {})
        status = "connected" if device in connected else "disconnected" if config else "missing"
        caption = campaign.line_roles[role]["name"] if campaign and role in campaign.line_roles else ROLES[role].label if role in ROLES else role
        name = config.get("name", device)
        assignments.append((caption, f"{device} · {name} · {status}", context.get("resolution", {}).get("origins", {}).get(role, label)))
        if calibration := config.get("pump_calibration"):
            assignments.append((f"{caption} calibration", calibration.get("source_name", "Loaded"), "Control Panel"))
        if calibration := config.get("seed_calibration"):
            assignments.append((f"{caption} calibration", f"{calibration.get('source_name', 'Loaded')} · {calibration['rpm']:g} RPM", "Control Panel"))
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
    resolved = {}
    if simulation:
        _, exp, values, _, mesh = model_inputs(simulation["seeded"])
        values = {**values, "InitialConcentration": exp["ConcentrationData"][1][0]}
        mixed_seed = any(exp.imed_simulation["seeded"] != simulation["seeded"] for exp in experiments)
        if mixed_seed:
            values["SeedMass"] = f"0 / {model_inputs(True)[2]['SeedMass']:g} · depends on starting profile"
    else:
        values = starting_process_defaults(setup, case)
        process_labels["InitialVolume_ml"] = PREVIEW_AMOUNTS["InitialVolume_ml"]
        try:
            resolved = starting_input_rows(setup, context)
        except ValueError:
            # Invalid or stale stage selections are described in the period rows below.
            pass
    for key, title in process_labels.items():
        if key == "PowerRatio" and backend == "imed":
            inputs.append((title, "Calculated per experiment from averages" if setup["mixing"]["enabled"] else 1., "Average mixing option"))
            continue
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
        current = values.get(key)
        if key in PREVIEW_AMOUNTS and resolved:
            readings = [row[0].get(key) for row in resolved.values()]
            if all(isinstance(value, Real) for value in readings):
                low, high = min(readings), max(readings)
                current = low if low == high else f"{low:.8g}–{high:.8g}"
                sources = {row[1].get(key, origin) for row in resolved.values()}
                origin = sources.pop() if len(sources) == 1 else "Per-run process inputs"
        inputs.append((title, current, origin))
    if not simulation:
        for name, overrides in setup["initial_processes"].items():
            for key, value in overrides.items():
                inputs.append((f"{name} · {process_labels.get(key, key)}", value, "Starting experiment override"))
        preparation = []
        for name, (values, _, _, _) in resolved.items():
            if charges := values.get("PreparationCharges_g"):
                detail = ", ".join(f"{role} {mass:.6g} g" for role, mass in charges.items())
                preparation.append((name, detail, "Preparation pump profiles"))
        if preparation:
            groups["Calibrated preparation charges"] = preparation

    recorded = groups["Recording"] = []
    period = groups["Modeling experiment period"] = []
    selected_campaign = None
    try:
        selected_campaign = modeling_campaign(campaign, setup)
        for exp in selected_campaign.experiments if selected_campaign else ():
            window = exp.model_window
            if window is None:
                window = (recording["start_s"], recording["end_s"] or max(p.points[-1].time_s for p in exp.profiles))
            names = [s.name for s in exp.run_stages if window[0] <= s.start_s and s.end_s <= window[1]]
            period.append((exp.name, f"{window[0]:g}–{window[1]:g} s · " + (", ".join(names) or "time window"),
                           "Experiment period"))
    except ValueError as error:
        period.append(("Check selected stages", str(error), "Experiment period"))
    if simulation:
        windows = sorted({(exp.imed_simulation["start_s"], exp.imed_simulation["end_s"]) for exp in experiments})
        recorded.append(("Crystallization windows (s)", ", ".join(f"{start:g}–{end:g}" for start, end in windows), "PBM simulation"))
        mapping = {**simulation["monitors"], "Temperature": f"{roles.get('T', '')}.temp_external"}
    else:
        recorded.extend((
            ("Concentration multiplier", recording["concentration_multiplier"], "Recording tab"),
            ("No antisolvent feed", recording["no_feed"], "Recording tab"),
            ("Antisolvent scale direction", "Decreasing" if recording["scale_direction"] == -1 else "Increasing", "Recording tab"),
            ("Scale median window (samples)", recording["smoothing"], "Recording tab"),
        ))
        available = {device: configs[device] for device in {*roles.values(), *monitors} if device in configs}
        try:
            required = imed_objective_channels(imed_response_weights(case)) if backend == "imed" else None
            mapping = (recording_mapping(setup, roles, available, required_measurements=required)
                       if "T" in roles else setup["channels"])
        except ValueError as error:
            recorded.append(("Check channel mappings", str(error), "Recording tab"))
            mapping = setup["channels"]
        else:
            if selected_campaign and "T" in roles:
                try:
                    source = antisolvent_recording_source(recording, mapping, selected_campaign,
                                                          roles, available, infer_no_feed=backend == "imed")
                except ValueError as error:
                    source = str(error)
                recorded.append(("AS feed measurement", source, "Selected experiment period"))
    for role in CHANNELS:
        inherited = "PBM simulation" if simulation else "Recording override" if role in setup["channels"] else "Automatic mapping"
        recorded.append((role, mapping.get(role) or "Not recorded", inherited))
    if not simulation:
        recorded.append(("Seed D10 / D50 / D90 (µm)", recording["seed_percentiles"], "Process inputs"))

    support = groups["Follow-up support"] = []
    if campaign and (campaign.line_roles or experiments[0].run_stages):
        support.append(("Charging and cleanup", "Inherited from the starting timeline", "Experiment Definition"))
    for role, profile in setup["support_profiles"].items():
        times, targets = profile["points"]
        support.append((ROLES[role].label, f"{len(times)} points · {min(targets):g}–{max(targets):g} {ROLES[role].unit}",
                        "Support profile override"))
    pump = setup["seed_pump"]
    seed_config = configs.get(roles.get("seed"), {})
    seed_curve = seed_config.get("seed_calibration")
    if seed_curve and (not pump or pump.get("use_device_calibration")):
        support.append(("Calibrated dry-seed delivery", f"{seed_curve['rpm']:g} RPM · up to {seed_curve['points'][-1][1]:g} g",
                        seed_curve["source_name"]))
    elif pump:
        try:
            from experiment.pump_calibration_files import seed_delivery_rate
            rate = seed_delivery_rate(seed_config, pump)
            support.append(("Calibrated seed delivery", f"{pump['rpm']:g} RPM · {rate:g} g/s",
                            "Loaded seed pump calibration" if pump.get("use_device_calibration") else "Devices"))
        except (ValueError, KeyError, TypeError) as error:
            support.append(("Check seed calibration", str(error), "Devices"))

    model = groups["Modeling"] = []
    if backend == "bo" and (bo := context["bo"]) is not None:
        model.append(("Selected prior observations", len(bo["selected"]), "BO page"))
        if bo.get("batch_horizon_s") is not None:
            model.append(("Batch horizon (s)", selected_horizon(bo), "BO Physical profiles override"))
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
        mixing = setup["mixing"]
        model.append(("Average mixing power", "Calculated per experiment from average RPM and volume" if mixing["enabled"] else "PowerRatio = 1", "IMED design"))
        if mixing["enabled"]:
            model.extend((("Impeller power number", mixing["power_number"], "Campaign constant"),
                          ("Impeller diameter (mm)", mixing["diameter_m"] * 1000, "Campaign constant"),
                          ("Volume basis", mixing["volume_ml"] if mixing["volume_mode"] == "fixed" else "Charges + recorded AS additions", "IMED design")))
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
             imed_response_weights(case), "IMED objectives"),
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
    if context.get("final_size_samples") is not None:
        model.append(("Final size samples to average", context["final_size_samples"], "Settings"))
    return groups
