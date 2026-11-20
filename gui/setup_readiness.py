"""Read-only setup checks; never connect devices or execute calculations."""
from pathlib import Path

import numpy as np

from gui.setup_checklist import SetupItem


def _valid_models(models):
    if not models:
        return False
    try:
        for model in models:
            values = np.asarray(model.get("paraset", []), dtype=float)
            if values.shape != (35, 4) or np.isnan(values).any() or np.isposinf(values).any():
                return False
            active = values[values[:, 0] == 1, 1:]
            if not len(active) or not np.isfinite(active).all() or np.any(active[:, 1] > active[:, 2]):
                return False
    except (TypeError, ValueError):
        return False
    return True


def _valid_solubility(coefficients):
    try:
        return isinstance(coefficients, dict) and all(np.isfinite(float(coefficients[key])) for key in "abcdefgh")
    except (KeyError, TypeError, ValueError):
        return False


def loop_readiness(setup, backend, context):
    from experiment.adaptive_loop import (
        IMED_OBJECTIVES, _number, _process, antisolvent_recording_source,
        imed_design_conditions, imed_objective_channels, imed_response_weights, recording_mapping,
    )
    from experiment.doe_decisions import starting_run_process
    from experiment.loop_process import starting_process_defaults
    from experiment.run_stages import modeling_campaign
    from imed_python.design import recipe_space

    items = []
    def check(key, title, task):
        try:
            detail = task()
        except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
            items.append(SetupItem(key, title, False, str(error)))
        else:
            items.append(SetupItem(key, title, True, detail))

    campaign = context.get("campaign")
    configs, roles = context.get("configs", {}), context.get("roles", {})
    monitors, connected = set(context.get("monitors", ())), set(context.get("connected", ()))
    case = context.get("case", {})
    effective = context.get("effective_setup", setup)
    from imed_python.demo_campaign import legacy_simulation

    simulation = legacy_simulation(campaign.experiments[0].imed_simulation) if campaign and campaign.experiments else None

    def batch():
        if not campaign or not campaign.experiments:
            raise ValueError("Generate and assign a Settings DoE, or load Control Panel experiments. Choose Starting batch under Used values.")
        return f"{len(campaign.experiments)} starting experiments · {context.get('label', campaign.name)}."
    check("batch", "Starting batch", batch)

    def folders():
        roots = context.get("folders", ())
        if len(roots) != 2 or any(not str(p).strip() or not Path(p).is_dir() for p in roots):
            raise ValueError("Choose existing experiment and model output folders in Settings → Files & presets.")
        return "Both selected output destinations exist."
    check("folders", "Output folders", folders)

    resolved = None
    def period():
        nonlocal resolved
        if not campaign:
            raise ValueError("Choose a starting batch before selecting its experiment period.")
        resolved = modeling_campaign(campaign, effective)
        if not resolved:
            raise ValueError("Select run stages or choose Manual start / end under Experiment period.")
        if any(exp.model_window is None for exp in resolved.experiments):
            # Legacy profiles have no named stages: the execution path uses the
            # configured recording offsets (or the validation simulator window).
            return "Profiles without named stages use the saved timing offsets. To edit them, choose Manual start / end under Experiment period. The PBM validation simulator uses its predefined period."
        return "The selected continuous experiment period resolves for every starting run; other stages remain recorded."
    check("period", "Experiment period", period)

    def devices():
        if "T" not in roles:
            raise ValueError("Assign the reactor temperature controller through MSI or Devices.")
        required = set(roles.values()) | monitors
        if campaign:
            required |= {p.device_id for exp in campaign.experiments for p in exp.profiles}
        missing = required - connected
        if missing:
            raise ValueError("Connect assigned devices and include required read-only monitors: " + ", ".join(sorted(missing)))
        messages = context.get("resolution", {}).get("messages", [])
        if messages:
            raise ValueError(" ".join(messages))
        return f"{len(required)} assigned devices/monitors connected. Review exact roles under Used values."
    check("devices", "Devices & monitors", devices)

    weights = []
    if backend == "imed":
        def objectives():
            nonlocal weights
            weights = imed_response_weights(case)
            enabled = [f"{label} ({weight:g})" for (label, _), weight in zip(IMED_OBJECTIVES, weights)
                       if weight > 0]
            if not enabled:
                raise ValueError("Open IMED objectives and set at least one positive weight for FBRM, concentration or D-values.")
            return "Selected objectives · " + ", ".join(enabled)
        check("pe", "IMED objectives", objectives)

    mapping = {}
    feed_source = None
    def channels():
        nonlocal mapping, feed_source
        if simulation:
            return "Built-in PBM simulator supplies the configured measurement channels."
        if "T" not in roles:
            raise ValueError("Assign temperature before choosing recording channels.")
        selected = {key: configs[key] for key in set(roles.values()) | monitors if key in configs}
        objective_channels = imed_objective_channels(weights) if backend == "imed" else None
        mapping = recording_mapping(effective, roles, selected, required_measurements=objective_channels)
        required = ["Temperature"]
        if backend == "bo":
            bo = context.get("bo") or {}
            outcomes = {key for key, target in bo.get("targets", {}).items() if target["enabled"] and target["weight"] > 0}
            outcomes |= {key for key, bounds in bo.get("output_bounds", {}).items() if any(v is not None for v in bounds)}
            if outcomes & {"d10", "d50", "d90"}:
                required += ["D10", "D50", "D90"]
            if "yield" in outcomes and bo.get("yield_basis") == "actual":
                required.append("Concentration")
        elif backend == "imed":
            required.extend(objective_channels)
        missing = [name for name in required if not mapping.get(name)]
        if missing:
            raise ValueError("Select recorded channels for " + ", ".join(missing) + " in Recording; include their devices in the campaign.")
        feed_source = antisolvent_recording_source(effective["recording"], mapping, resolved, roles, selected,
                                                   infer_no_feed=backend == "imed")
        return "Required channels are available. Antisolvent input · " + feed_source + "."
    check("recording", "Recording channels", channels)

    def inputs():
        if simulation:
            return "Starting process inputs and seed distribution come from the PBM simulation."
        if backend == "imed" and effective["mixing"]["enabled"] and "stirring_rate" not in roles:
            raise ValueError("Assign the main stirrer through MSI or Devices to calculate average mixing power.")
        if not campaign:
            raise ValueError("Choose the starting batch to resolve process inputs per experiment.")
        common = starting_process_defaults(effective, case)
        concentration = effective["recording"]["initial_concentration"]
        if concentration is not None:
            common.setdefault("InitialConcentration", concentration)
        input_campaign = resolved or modeling_campaign(campaign, effective)
        for exp in input_campaign.experiments:
            try:
                values = _process(starting_run_process(exp, effective, roles, configs, input_campaign.line_roles, common))
                if "InitialConcentration" in values:
                    _number(values["InitialConcentration"], "initial concentration", 0, 1 - 1e-12)
                elif not mapping.get("Concentration"):
                    raise ValueError("Enter Initial concentration or map a concentration channel sampled at the start.")
            except ValueError as error:
                raise ValueError(f"{exp.name}: {error}") from error
            if values["SeedMass"] > 0 and (len(effective["recording"]["seed_percentiles"]) != 3
                                           or min(effective["recording"]["seed_percentiles"]) <= 0):
                raise ValueError("Positive seed mass requires positive seed D10/D50/D90 under Process inputs.")
        return "Starting masses, densities and seed information resolve for every experiment; Used values shows their sources."
    check("inputs", "Process inputs", inputs)

    bo = context.get("bo") or {}
    yield_target = bo.get("targets", {}).get("yield", {})
    yield_bounds = bo.get("output_bounds", {}).get("yield", ())
    theoretical_yield = backend == "bo" and bo.get("yield_basis") == "theoretical" and (
        yield_target.get("enabled") and yield_target.get("weight", 0) > 0 or any(v is not None for v in yield_bounds))
    if backend == "imed" or theoretical_yield:
        def solubility():
            if not simulation and not _valid_solubility(case.get("Solubility")):
                raise ValueError("Open Solubility and enter coefficients a–h before starting the loop.")
            return "Solubility coefficients are available."
        check("solubility", "Solubility", solubility)

    def model():
        if backend == "bo":
            if not bo or not any(t["enabled"] and t["weight"] > 0 for t in bo["targets"].values()):
                raise ValueError("Open BO targets, bounds and profiles and enable at least one target with a positive weight.")
            return "At least one BO target is enabled with a positive weight."
        if not simulation and not _valid_models(case.get("ModelParameter")):
            raise ValueError("Add a candidate kinetic model with estimated parameters and valid bounds in IMED settings.")
        return "Candidate models are available for parameter estimation."
    check("model", "BO targets" if backend == "bo" else "IMED models", model)
    if backend == "imed":
        def design():
            if (effective["imed"]["use_doe_decisions"] and (decisions := context.get("doe_decisions"))
                    and decisions["traces"].get("temperature")):
                from experiment.doe_decisions import decision_groups, decision_summary

                groups = decision_groups(decisions, imed=True)
                return f"DoE design inputs: {decision_summary(decisions, groups)}. Stirring also requires a fitted power term."
            conditions = effective["imed"]["conditions"]
            if not conditions:
                raise ValueError("Define the follow-up experiment bounds under IMED design.")
            recipe_space(imed_design_conditions(conditions, no_feed=effective["recording"]["no_feed"]
                                                or feed_source == "No antisolvent feed"))
            return "IMED follow-up experiment conditions are valid."
        check("design", "IMED design bounds", design)
    items.append(SetupItem("limits", "Stopping rule", True,
                           f"Stop after {effective['max_followups']} recommended experiments after the initial batch. Dashboard Stop remains available."))
    advanced = bool(effective["initial_processes"] or effective["support_profiles"] or effective["manual_simulation"])
    items.append(SetupItem("advanced", "Advanced overrides", advanced,
                           "Optional named process overrides, support profiles and simulated PBM telemetry. Review all active overrides here.", required=False))
    return items


def imed_readiness(case):
    from experiment.adaptive_loop import imed_response_weights

    experiments, processes, models = (case.get(key, []) for key in ("ExpData", "ProcInf", "ModelParameter"))
    def populated(value):
        return value is not None and (np.asarray(value).size > 0 if not isinstance(value, dict) else bool(value))
    solubility = case.get("Solubility", {})
    valid_solubility = _valid_solubility(solubility)
    def valid_trace(exp):
        try:
            points = np.asarray(exp.get("TemperatureData", []), dtype=float)
            return bool(points.ndim == 2 and points.shape[0] == 2 and points.shape[1] >= 2
                        and np.isfinite(points).all() and np.all(np.diff(points[0]) > 0))
        except (TypeError, ValueError):
            return False
    data_ready = bool(experiments) and all(valid_trace(exp) for exp in experiments)
    process_ready = bool(experiments) and len(processes) == len(experiments)
    if process_ready:
        from experiment.adaptive_loop import _process
        try:
            for process in processes:
                _process(process)
        except (ValueError, TypeError, KeyError):
            process_ready = False
    model_ready = _valid_models(models)
    weights = case.get("peweights", case.get("objweightfactor", {}))
    weights = weights if isinstance(weights, dict) else {}
    response_detail = "Set response weights in IMED objectives. Open PE settings to review the grid and optimizer."
    objectives_ready = False
    try:
        values = imed_response_weights(case)
        objectives_ready = weighted = any(values)
        if weighted and data_ready and process_ready and valid_solubility:
            from imed_python.estimation import FitObjective
            from imed_python.pe_settings import effective_pe_pbm
            FitObjective(experiments, processes, solubility, effective_pe_pbm(case), values,
                         weights.get("Terms", "CSD"))
    except (ValueError, TypeError, KeyError, IndexError) as error:
        weighted = False
        response_detail = f"Parameter Estimation needs attention: {error}"
    fitted = bool(models) and any(populated(model.get("para", [])) for model in models)
    return [
        SetupItem("experiments", "Experimental data", data_ready, f"{len(experiments)} experiments. Add or import temperature profiles and measured responses in Edit experimental inputs."),
        SetupItem("process", "Process per experiment", process_ready, "Each experiment needs its corresponding starting masses, densities, shape factor and seed amount in Edit experimental inputs."),
        SetupItem("solubility", "Solubility", valid_solubility, "Open Solubility to enter coefficients a–h and preview the curve."),
        SetupItem("objectives", "IMED objectives", objectives_ready, "Open IMED objectives and set at least one positive weight."),
        SetupItem("models", "Candidate models", model_ready, "Generate a candidate model or review its 35 parameter definitions under Model Configurations."),
        SetupItem("pe", "PE responses & settings", weighted, response_detail),
        SetupItem("fits", "Fitted results", fitted, "Run Parameter Estimation after completing the inputs. Fitted models enable subsequent design and uncertainty work.", required=False),
    ]
