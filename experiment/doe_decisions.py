"""Decision variables supported by the starting experiments, in process units."""
from collections.abc import Mapping
from dataclasses import asdict

import numpy as np


LABELS = {"temperature": "Temperature profile", "feed": "Antisolvent profile",
          "seed": "Seed mass", "stirring": "Stirring rate", "volume": "Starting liquid volume"}
TRACE_GROUPS = ("temperature", "feed", "stirring")
SCALAR_GROUPS = ("seed", "volume")


def charge_volume(process):
    from bo.profiles import scalar

    volume = scalar(process.get("InitialVolume_ml"))
    if volume is not None and volume > 0:
        return volume
    fields = [scalar(process.get(key)) for key in ("SolvMass", "AntiSolvMass", "SolventDensity", "AntiSolventDensity")]
    if all(value is not None for value in fields):
        solvent, anti, density, anti_density = fields
        if solvent > 0 and anti >= 0 and density > 0 and anti_density > 0:
            return solvent / density + anti / anti_density
    return None


def _same_trace(left, right, tolerance):
    # Differences on a linear interval reach their extrema at its endpoints.
    # Also check the left limit at every step transition.
    knots = np.union1d(left.time, right.time)
    if left.step or right.step:
        knots = np.r_[knots, np.nextafter(knots[1:], -np.inf)]
    return bool(np.allclose(left.sample(knots), right.sample(knots), atol=tolerance, rtol=0))


def capture_decisions(runs):
    """Keep distinct commanded waveforms and the observed scalar levels."""
    from bo.decoder import fixed_horizon
    from bo.profiles import CHANNELS, make_profile

    space = {"traces": {}, "levels": {}}
    for group in TRACE_GROUPS:
        entries = [run["profiles"].get(group) for run in runs]
        if not entries or not all(entries):
            continue
        commanded = all("command" in entry["source"].casefold() for entry in entries)
        tolerance = 1e-7 if commanded else CHANNELS[group][2]
        profiles = [make_profile(entry["points"], entry["source"], step=entry["step"]) for entry in entries]
        if any(profile is None or len(profile.time) < 2 or profile.time[0] != 0
               or (group != "temperature" and np.any(profile.values < 0)) for profile in profiles):
            continue
        if group == "feed":
            # Shorter recipes stop feeding when their modeled period ends.
            end = max(profile.time[-1] for profile in profiles)
            profiles = [fixed_horizon(profile, end, stop=True) for profile in profiles]
        distinct = []
        for profile in profiles:
            if not any(_same_trace(profile, prior, tolerance) for prior in distinct):
                distinct.append(profile)
        if distinct:
            space["traces"][group] = [profile.payload() for profile in distinct]
        if group == "stirring":
            # BO uses the waveform. The PBM uses a single time-weighted RPM.
            levels = [float(profile.total() / (profile.time[-1] - profile.time[0])) for profile in distinct
                      if len(profile.time) > 1 and profile.time[-1] > profile.time[0]]
            if levels:
                space["levels"][group] = _distinct_levels(levels, tolerance)
    for group in SCALAR_GROUPS:
        entries = [run["profiles"].get(group) for run in runs]
        if entries and all(entries):
            space["levels"][group] = _distinct_levels([entry["points"][1][0] for entry in entries], CHANNELS[group][2])
    return space


def _distinct_levels(values, tolerance):
    levels = []
    for value in sorted(map(float, values)):
        if not levels or value - levels[-1] > max(1e-8, tolerance):
            levels.append(value)
    return levels


def decision_groups(space, *, imed=False):
    if not space:
        return set()
    traces, levels = space.get("traces", {}), space.get("levels", {})
    groups = {key for key, choices in traces.items() if len(choices) > 1}
    if imed:
        groups.discard("stirring")
    groups.update(key for key, values in levels.items() if len(values) > 1)
    return groups


def validate_decisions(space):
    """Accept plain data only so the selection round-trips in setup and case files."""
    from bo.profiles import pairs

    if space is None:
        return None
    if isinstance(space, str):
        import json

        space = json.loads(space)
    if not isinstance(space, dict) or set(space) != {"traces", "levels"}:
        raise ValueError("Invalid saved DoE decision variables.")
    traces, levels = space["traces"], space["levels"]
    if not isinstance(traces, Mapping) or not isinstance(levels, Mapping):
        raise ValueError("Invalid saved DoE decision variables.")
    for group, entries in traces.items():
        if group not in TRACE_GROUPS or not isinstance(entries, list) or not entries:
            raise ValueError("Invalid saved DoE decision profiles.")
        for entry in entries:
            points = pairs(entry.get("points")) if isinstance(entry, dict) else None
            if (points is None or points.shape[1] < 2 or points[0, 0] != 0
                    or type(entry.get("step")) is not bool or not isinstance(entry.get("source"), str)
                    or (group != "temperature" and np.any(points[1] < 0))):
                raise ValueError(f"Invalid saved DoE {LABELS[group].lower()}.")
    for group, values in levels.items():
        if (group not in (*SCALAR_GROUPS, "stirring") or not isinstance(values, list) or not values
                or any(type(value) not in (float, int) or not np.isfinite(value) or value < 0 for value in values)
                or (group == "volume" and min(values) <= 0)
                or any(left >= right for left, right in zip(values, values[1:]))):
            raise ValueError("Invalid saved DoE levels.")
    return space


def case_decisions(case, folder=None):
    """Use the saved starting batch, otherwise infer from the loaded experiments."""
    from pathlib import Path
    from bo.profiles import case_runs
    from imed_python.campaign_import import read_json

    saved = case.get("PythonWorkflow", {}).get("doe_decisions")
    if saved is None and folder is not None:
        definition = Path(folder) / "definition.json"
        if definition.is_file():
            saved = read_json(definition).get("campaign", {}).get("doe_decisions")
    if saved is not None:
        saved = validate_decisions(saved)
        if folder is None:
            return saved
        # A saved plan is authoritative where the setup could resolve it. At
        # import the recording can fill channels that needed measured delivery.
        actual = capture_decisions(case_runs(case, folder, prefer_commands=True))
        return {key: {**actual[key], **saved[key]} for key in saved}
    experiments = case.get("ExpData", [])
    if len(experiments) < 2:
        return None
    starting = [index for index, experiment in enumerate(experiments) if not experiment.get("RecipeStage")]
    initial_case = case
    if starting and len(starting) != len(experiments):
        processes = case.get("ProcInf", [])
        initial_case = {**case, "ExpData": [experiments[index] for index in starting],
                        "ProcInf": [processes[index] if index < len(processes) else {} for index in starting]}
    from bo.profiles import pairs

    if not any(pairs(exp.get("TemperatureData")) is not None for exp in initial_case["ExpData"]):
        return None
    return capture_decisions(case_runs(initial_case, folder, prefer_commands=True))


def starting_run_process(experiment, setup, roles, configs, line_roles, defaults, *, sources=None, issues=None):
    from experiment.material_balance import initial_charges
    from imed_python.demo_campaign import legacy_simulation, model_inputs

    if simulation := legacy_simulation(experiment.imed_simulation):
        process = dict(model_inputs(simulation["seeded"])[2])
        if (volume := charge_volume(process)) is not None:
            process["InitialVolume_ml"] = volume
        if sources is not None:
            sources.update(dict.fromkeys(process, "PBM simulation"))
        return process
    overrides = setup["initial_processes"].get(experiment.name, {})
    process = dict(defaults)
    if experiment.seed_mass_g is not None:
        process["SeedMass"] = experiment.seed_mass_g
        if sources is not None:
            sources["SeedMass"] = "Defined seed dose"
    elif "SeedMass" not in overrides and experiment.model_window and experiment.model_window[0] > 0:
        seed_devices = {device for role, device in roles.items() if line_roles.get(role, {}).get("role", role) == "seed"}
        device = next(iter(seed_devices)) if len(seed_devices) == 1 else None
        config = configs.get(device, {})
        profile = next((p for p in experiment.profiles if p.device_id == device), None)
        if config.get("seed_calibration") and profile is not None and any(p.target > 0 for p in profile.points):
            from experiment.pump_calibration_files import seed_mass_from_profile

            try:
                process["SeedMass"] = seed_mass_from_profile(profile, config, experiment.model_window[0])
            except ValueError as error:
                if issues is None:
                    raise
                issues["SeedMass"] = str(error)
                process.pop("SeedMass", None)
            else:
                if (issues is not None and process["SeedMass"] == 0
                        and any(p.target > 0 and experiment.model_window[0] <= p.time_s < experiment.model_window[1]
                                for p in profile.points)):
                    issues["SeedMass"] = "The seed pulse starts in the modeled period. Select a period after seeding to infer its starting dose."
                    process.pop("SeedMass")
                elif sources is not None:
                    sources["SeedMass"] = "Dry-seed pump calibration"
        elif (issues is not None and profile is not None and any(p.target > 0 for p in profile.points)
              and not process.get("SeedMass") and "SeedMass" not in setup["process"]):
            issues["SeedMass"] = f"Load a dry-seed calibration for {device} or enter the known dry seed mass. Seeding belongs before the experiment period."
            process.pop("SeedMass", None)
    process.update(overrides)
    if {"SolvMass", "AntiSolvMass", "SolventDensity", "AntiSolventDensity"} & overrides.keys() and "InitialVolume_ml" not in overrides:
        process.pop("InitialVolume_ml", None)
    if setup["derive_initial_charges"]:
        process = initial_charges(experiment, process, roles, configs, overrides=overrides, line_roles=line_roles, issues=issues)
        if sources is not None:
            charges = process.get("PreparationCharges_g", {})
            for key, materials in (("SolvMass", {"solvent", "solution"}), ("AntiSolvMass", {"AS"})):
                if materials & charges.keys() and key in process:
                    sources[key] = "Preparation pump profiles"
    if (volume := charge_volume(process)) is not None:
        process["InitialVolume_ml"] = volume
        if sources is not None:
            sources["InitialVolume_ml"] = "Starting masses and densities"
    if sources is not None:
        sources.update(dict.fromkeys(overrides, "Starting experiment override"))
    return process


def planned_decisions(campaign, setup, roles, configs, case):
    from bo.profiles import Profile, command_profiles, scalar
    from experiment.loop_process import starting_process_defaults
    from experiment.run_stages import modeling_campaign

    campaign = modeling_campaign(campaign, setup)
    if campaign is None or not campaign.experiments:
        return None
    definition = {"devices": configs, "campaign": {"line_roles": campaign.line_roles}}
    has_antisolvent = "AS" in roles or any(lane.get("role") == "AS" and key in roles
                                         for key, lane in campaign.line_roles.items())
    defaults = starting_process_defaults(setup, case)
    recording = setup["recording"]
    if recording["initial_concentration"] is not None:
        defaults.setdefault("InitialConcentration", recording["initial_concentration"])
    runs = []
    for experiment in campaign.experiments:
        process = starting_run_process(experiment, setup, roles, configs, campaign.line_roles, defaults)
        end = max(profile.points[-1].time_s for profile in experiment.profiles)
        window = experiment.model_window
        if experiment.imed_simulation is not None:
            window = (experiment.imed_simulation["start_s"], experiment.imed_simulation["end_s"])
        start, end = window or (recording["start_s"], recording["end_s"] or end)
        if end <= start:
            raise ValueError(f"{experiment.name} has no positive modeling period.")
        profiles = command_profiles(asdict(experiment), process, definition, start, end - start, roles)
        if recording["no_feed"] or not has_antisolvent:
            profiles["feed"] = Profile(np.array([0., end - start]), np.zeros(2), "Command · No AS feed")
        for key, value in (("seed", scalar(process.get("SeedMass"))), ("volume", charge_volume(process))):
            if value is not None:
                profiles[key] = Profile(np.zeros(1), np.array([value]), "Starting process setting")
        runs.append({"profiles": {key: value.payload() for key, value in profiles.items()}})
    return capture_decisions(runs)


def decision_summary(space, groups=None):
    groups = decision_groups(space) if groups is None else set(groups)
    return ", ".join(label for group, label in LABELS.items() if group in groups) or "None"
