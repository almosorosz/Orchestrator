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


def loop_readiness(setup, backend, context):
    from experiment.adaptive_loop import _process, recording_mapping
    from experiment.loop_process import starting_process_defaults
    from experiment.material_balance import initial_charges
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
    simulation = campaign.experiments[0].imed_simulation if campaign and campaign.experiments else None

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

    def period():
        if not campaign:
            raise ValueError("Choose a starting batch before selecting its experiment period.")
        resolved = modeling_campaign(campaign, effective)
        if not resolved:
            raise ValueError("Select the experiment stages, or enter manual start/end times in Recording.")
        if any(exp.model_window is None for exp in resolved.experiments):
            # Legacy profiles have no named stages: the execution path uses the
            # configured recording offsets (or the validation simulator window).
            return "Profiles without named stages use the Recording start/end offsets; the PBM validation simulator uses its predefined crystallization period."
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

    mapping = {}
    def channels():
        nonlocal mapping
        if simulation:
            return "Built-in PBM simulator supplies the configured measurement channels."
        if "T" not in roles:
            raise ValueError("Assign temperature before choosing recording channels.")
        selected = {key: configs[key] for key in set(roles.values()) | monitors if key in configs}
        mapping = recording_mapping(effective, roles, selected)
        required = ["Temperature"]
        if effective["recording"]["initial_concentration"] is None:
            required.append("Concentration")
        if backend == "bo":
            bo = context.get("bo") or {}
            outcomes = {key for key, target in bo.get("targets", {}).items() if target["enabled"] and target["weight"] > 0}
            outcomes |= {key for key, bounds in bo.get("output_bounds", {}).items() if any(v is not None for v in bounds)}
            if outcomes & {"d10", "d50", "d90"}:
                required += ["D10", "D50", "D90"]
            if "yield" in outcomes and bo.get("yield_basis") == "actual":
                required.append("Concentration")
        elif backend == "imed":
            weights = case.get("peweights", case.get("objweightfactor", {}))
            weights = weights if isinstance(weights, dict) else {}
            for index, default, fields in ((1, 0, ["FBRM"]), (2, 1, ["Concentration"]), (3, 1, ["D10", "D50", "D90"])):
                if float(weights.get(f"W{index}", default)) > 0:
                    required += fields
        missing = [name for name in required if not mapping.get(name)]
        if missing:
            raise ValueError("Select recorded channels for " + ", ".join(missing) + " in Recording; include their devices in the campaign.")
        if not effective["recording"]["no_feed"] and not mapping.get("AS scale"):
            pump = configs.get(roles.get("AS"), {})
            if pump.get("type") != "ismatec" and not pump.get("pump_calibration"):
                raise ValueError("Map an AS scale or an assigned AS pump with a loaded mass calibration.")
        return "Required measurement channels are available in the assigned campaign."
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
        for exp in campaign.experiments:
            override = effective["initial_processes"].get(exp.name, {})
            values = {**common, **override}
            if effective["derive_initial_charges"]:
                values = initial_charges(exp, values, roles, configs, overrides=override, line_roles=campaign.line_roles)
            values = _process(values)
            if values["SeedMass"] > 0 and (len(effective["recording"]["seed_percentiles"]) != 3
                                           or min(effective["recording"]["seed_percentiles"]) <= 0):
                raise ValueError("Positive seed mass requires positive seed D10/D50/D90 under Process inputs.")
        return "Starting masses, densities and seed information resolve for every experiment; Used values shows their sources."
    check("inputs", "Process inputs", inputs)

    def model():
        if backend == "bo":
            bo = context.get("bo")
            if not bo or not any(t["enabled"] and t["weight"] > 0 for t in bo["targets"].values()):
                raise ValueError("Open BO → Input bounds and targets and enable at least one target with a positive weight.")
            if bo["targets"]["yield"]["enabled"] and bo["yield_basis"] == "theoretical" and not case.get("Solubility") and not simulation:
                raise ValueError("Theoretical yield needs solubility from IMED. Load it in IMED or choose actual yield on the BO page.")
            return "BO has an enabled objective. Review its bounds and constraints on the BO page."
        if not simulation and (not _valid_models(case.get("ModelParameter")) or not case.get("Solubility")):
            raise ValueError("Open IMED and define candidate models and solubility before starting the loop.")
        return "Candidate models/solubility are available; the loop will estimate parameters from the new recordings."
    check("model", "BO targets" if backend == "bo" else "IMED models", model)
    if backend == "imed":
        def pe_settings():
            weights = case.get("peweights", case.get("objweightfactor", {}))
            weights = weights if isinstance(weights, dict) else {}
            values = [float(weights.get(f"W{i}", default)) for i, default in enumerate((0, 1, 1), 1)]
            if not np.isfinite(values).all() or min(values) < 0 or sum(value > 0 for value in values) < 2:
                raise ValueError("Automatic IMED needs at least two positive PE response weights to continue through D-optimal design. Set these in IMED → Parameter Estimation.")
            return "PE has at least two enabled response terms for the complete IMED loop."
        check("pe", "PE responses", pe_settings)
        def design():
            conditions = effective["imed"]["conditions"]
            if not conditions:
                raise ValueError("Define the follow-up experiment bounds under IMED design.")
            recipe_space(conditions)
            return "IMED follow-up experiment conditions are valid."
        check("design", "IMED design bounds", design)
    items.append(SetupItem("limits", "Stopping rule", True,
                           f"Stop after {effective['max_followups']} recommended experiments after the initial batch. Dashboard Stop remains available."))
    advanced = bool(effective["initial_processes"] or effective["support_profiles"] or effective["manual_simulation"])
    items.append(SetupItem("advanced", "Advanced overrides", advanced,
                           "Optional named process overrides, support profiles and simulated PBM telemetry. Review all active overrides here.", required=False))
    return items


def imed_readiness(case):
    experiments, processes, models = (case.get(key, []) for key in ("ExpData", "ProcInf", "ModelParameter"))
    def populated(value):
        return value is not None and (np.asarray(value).size > 0 if not isinstance(value, dict) else bool(value))
    solubility = case.get("Solubility", {})
    try:
        valid_solubility = isinstance(solubility, dict) and all(np.isfinite(float(solubility[key])) for key in "abcdefgh")
    except (KeyError, ValueError, TypeError):
        valid_solubility = False
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
    response_detail = "Select the measured responses and weights in Parameter Estimation. Review the solver, mesh and optimization limits there."
    try:
        values = [float(weights.get(f"W{i}", 0)) for i in (1, 2, 3)]
        weighted = bool(np.isfinite(values).all() and min(values) >= 0 and max(values) > 0)
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
        SetupItem("solubility", "Solubility", bool(valid_solubility), "Enter finite solubility coefficients a–h in Edit experimental inputs → Solubility."),
        SetupItem("models", "Candidate models", model_ready, "Generate a candidate model or review its 35 parameter definitions under Model Configurations."),
        SetupItem("pe", "PE responses & settings", weighted, response_detail),
        SetupItem("fits", "Fitted results", fitted, "Run Parameter Estimation after completing the inputs. Fitted models enable subsequent design and uncertainty work.", required=False),
    ]
