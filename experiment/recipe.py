"""Editable, device-lane recipes compiled into the shared campaign profiles."""
from copy import deepcopy
from math import isfinite

import numpy as np
import pandas as pd

from experiment.doe.doe_creator import ExperimentCreator
from experiment.doe_campaign import ROLES, ProfileRole
from experiment.run_stages import PURPOSES, recipe_run_stages, definition_window


TIME_UNITS = {"s": 1., "min": 60., "h": 3600.}
SYSTEM_FIELDS = ("v_heat_max", "v_cool_max", "AS_add_max", "T_room", "sol_add_max",
                 "time_to_wait_prior_exp", "aceton_boil_out_T", "wait_after_exp", "t_before_open_valve")


def parameter(value, mode="Fixed", minimum=None, maximum=None):
    return {"value": float(value), "mode": mode, "min": float(value if minimum is None else minimum),
            "max": float(value if maximum is None else maximum)}


def recipe_roles(recipe):
    return {lane["id"]: ProfileRole(lane["name"], ROLES[lane["role"]].unit, ROLES[lane["role"]].compatible_types)
            for lane in recipe["lanes"]}


def recipe_parameters(recipe):
    for stage in recipe["stages"]:
        yield f"{stage['id']}.duration", stage["duration"]
        for lane, action in stage["actions"].items():
            for key in (("start", "end") if action["kind"] == "Ramp" else ("end",)):
                yield f"{stage['id']}.{lane}.{key}", action[key]
    for lane in recipe["lanes"]:
        for section in lane.get("profile", []):
            yield f"profile:{lane['id']}:{section['id']}.duration", section["duration"]
            for key in (("start", "end") if section["kind"] == "Ramp" else ("end",)):
                yield f"profile:{lane['id']}:{section['id']}.{lane['id']}.{key}", section[key]


def sampled_parameter(item):
    return item["mode"] == "DoE" and item["min"] < item["max"] and not item.get("equal_previous", False)


def parameter_context(recipe, path):
    """Resolve an internal parameter key to the names and units shown in the editor."""
    lanes = {lane["id"]: lane for lane in recipe["lanes"]}
    sections = [(stage["id"], f'Stage "{stage["name"]}"', stage["actions"]) for stage in recipe["stages"]]
    sections.extend((f'profile:{lane["id"]}:{section["id"]}', f'Section "{section["name"]}"', {lane["id"]: section})
                    for lane in recipe["lanes"] for section in lane.get("profile", []))
    for prefix, location, actions in sections:
        if path == f"{prefix}.duration":
            unit = recipe.get("time_unit", "min")
            return f"{location} → Duration", unit, TIME_UNITS[unit], False
        for lane_id, action in actions.items():
            for key, title in (("start", "Start"), ("end", "End")):
                if path == f"{prefix}.{lane_id}.{key}":
                    lane = lanes[lane_id]
                    unit = "g" if action["kind"] == "Dose" else ROLES[lane["role"]].unit
                    field = "Mass addition" if action["kind"] == "Dose" else title + (" temperature" if lane["role"] == "T" else " target")
                    return f'{location} → {lane["name"]} → {field}', unit, 1., lane["role"] == "T"
    return "Experiment parameter", "", 1., False


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or recipe.get("version") != 1:
        raise ValueError("Choose an Experiment Definition recipe.")
    if recipe.get("time_unit", "min") not in TIME_UNITS:
        raise ValueError("Choose seconds, minutes or hours for the time display.")
    lanes, stages = recipe["lanes"], recipe["stages"]
    for entries, label in ((lanes, "device lines"), (stages, "stages")):
        independent = label == "stages" and any(lane.get("profile") for lane in lanes)
        if not isinstance(entries, list) or (not entries and not independent) or len({item["id"] for item in entries}) != len(entries):
            raise ValueError(f"Add distinct named {label}.")
        if any(not isinstance(item["id"], str) or not item["id"] or not str(item["name"]).strip() for item in entries):
            raise ValueError(f"Enter names for all {label}.")
    by_id = {lane["id"]: lane for lane in lanes}
    devices = [lane.get("device_id") for lane in lanes if lane.get("device_id")]
    if len(devices) != len(set(devices)):
        raise ValueError("Use one lane per physical device. Reuse it in several stages.")
    for lane in lanes:
        if lane["role"] not in ROLES or not isfinite(lane["initial"]):
            raise ValueError("Each line needs a supported device role and a finite initial value.")
        sections = lane.get("profile", [])
        if not isinstance(sections, list) or len({section["id"] for section in sections}) != len(sections):
            raise ValueError("Use distinct sections in each device profile.")
        if any(not isinstance(section["id"], str) or not section["id"] or not str(section["name"]).strip() for section in sections):
            raise ValueError("Enter names and IDs for every device profile section.")
        if sections and any(lane["id"] in stage["actions"] for stage in stages):
            raise ValueError(f"{lane['name']}: use either its independent profile or shared-stage actions.")
    all_stages = stages + [{"purpose": "Experiment", "actions": {lane["id"]: section}}
                           for lane in lanes for section in lane.get("profile", [])]
    stage_ids = {stage["id"]: index for index, stage in enumerate(stages)}
    for index, stage in enumerate(all_stages):
        if stage["purpose"] not in PURPOSES or set(stage["actions"]) - by_id.keys():
            raise ValueError("Each stage needs a valid purpose and existing device lines.")
        for lane_id, action in stage["actions"].items():
            role = by_id[lane_id]["role"]
            if not isinstance(action.get("until_end", False), bool):
                raise ValueError("Choose a valid end for each action span.")
            if action["kind"] not in {"Hold", "Ramp", "Dose"} or (role == "valve_state" and action["kind"] != "Hold"):
                raise ValueError("Valve actions must hold a binary state. Other devices support holds and ramps.")
            if action["kind"] == "Dose" and ROLES[role].unit != "g/h":
                raise ValueError("Mass additions need a solution, solvent, antisolvent or cleaning pump line.")
            if action["kind"] == "Dose":
                amount = action["end"]
                if not amount.get("equal_previous") and (amount["value"] if amount["mode"] == "Fixed" else amount["min"]) < 0:
                    raise ValueError("A mass addition must be nonnegative.")
            if role == "T" and action.get("control_mode") not in {"Jacket", "Process"}:
                raise ValueError("Choose jacket or internal vessel (Process) temperature control.")
            if action.get("until_stage") and (index >= len(stages) or action["until_stage"] not in stage_ids
                                               or stage_ids[action["until_stage"]] < index):
                raise ValueError("An action must end in its own stage or a later stage.")
            if index < len(stages):
                stop = len(stages) - 1 if action.get("until_end") else stage_ids.get(action.get("until_stage"), index)
                if any(lane_id in following["actions"] for following in stages[index + 1:stop + 1]):
                    raise ValueError("Overlapping actions on the same device line. Shorten the span or remove the later action.")
    for name, item in recipe_parameters(recipe):
        invalid_numbers = [key for key in ("value", "min", "max") if isinstance(item[key], bool)
                           or not isinstance(item[key], (int, float)) or not isfinite(item[key])]
        if item["mode"] not in {"Fixed", "DoE", "Alternate"} or invalid_numbers or item["min"] > item["max"]:
            context, unit, scale, temperature = parameter_context(recipe, name)
            if item["mode"] not in {"Fixed", "DoE", "Alternate"}:
                reason = f'Unknown Across experiments option {item["mode"]!r}. Choose Fixed, DoE or Alternate.'
            elif invalid_numbers:
                key = invalid_numbers[0]
                field = {"value": "Value", "min": "Minimum", "max": "Maximum"}[key]
                reason = f"{field} is {item[key]!r}; enter a finite number."
            else:
                reason = (f'Minimum {item["min"] / scale:g} {unit} is greater than Maximum {item["max"] / scale:g} {unit}. '
                          "Set Minimum less than or equal to Maximum.")
                if temperature:
                    reason += " For a cooling ramp, use Start and End for the temperature decrease; Minimum/Maximum define the DoE range."
            raise ValueError(f"{context}:\n{reason}")
        low = item["value"] if item["mode"] == "Fixed" else item["min"]
        high = item["value"] if item["mode"] == "Fixed" else item["max"]
        if name.endswith(".duration") and low <= 0:
            raise ValueError("Stage durations must be positive in every experiment.")
        lane_id = name.split(".")[1] if not name.endswith(".duration") else None
        if not isinstance(item.get("equal_previous", False), bool) or (item.get("equal_previous") and not lane_id):
            raise ValueError("Equal to previous is available for device values, not durations.")
        if lane_id and not item.get("equal_previous") and by_id[lane_id]["role"] == "valve_state" and (low not in (0, 1) or high not in (0, 1)
                                                                 or item["mode"] == "DoE"):
            raise ValueError("Valve states must be fixed at 0/1 or alternate between 0 and 1.")
    return recipe


def compile_recipe(recipe, values, system, *, check_limits=True, limit_warnings=None):
    def check_limit(message):
        if check_limits:
            raise ValueError(message)
        if limit_warnings is not None:
            limit_warnings.append(message)

    lanes = recipe["lanes"]
    times = np.r_[0., np.cumsum([values[f"{stage['id']}.duration"] for stage in recipe["stages"]])]
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Stage durations must produce an increasing finite timeline.")
    schedules = {}
    horizon = float(times[-1])
    for lane in lanes:
        sections = lane.get("profile", [])
        if sections:
            elapsed, schedule = 0., []
            for section in sections:
                prefix = f"profile:{lane['id']}:{section['id']}"
                end = elapsed + values[f"{prefix}.duration"]
                if not np.isfinite(end) or end <= elapsed:
                    raise ValueError("Device profile sections need positive finite durations.")
                schedule.append((elapsed, end, section, prefix, section["name"]))
                elapsed = end
            horizon = max(horizon, elapsed)
        else:
            schedule, skip = [], -1
            for index, stage in enumerate(recipe["stages"]):
                if index <= skip:
                    continue
                action = stage["actions"].get(lane["id"])
                skip = (len(recipe["stages"]) - 1 if action and action.get("until_end") else
                        next((i for i, item in enumerate(recipe["stages"]) if action and item["id"] == action.get("until_stage")), index))
                schedule.append((times[index], times[skip + 1], action, stage["id"], stage["name"]))
        schedules[lane["id"]] = schedule
    if horizon <= 0:
        raise ValueError("Add at least one stage or device profile section.")
    for lane in lanes:
        schedules[lane["id"]] = [(begin, horizon if action and action.get("until_end") else end, action, prefix, name)
                                  for begin, end, action, prefix, name in schedules[lane["id"]]]
    traces, modes = {}, []
    for lane in lanes:
        current = lane["initial"]
        x, y = [0.], [current]
        last_mode = None
        for begin, end, action, prefix, name in schedules[lane["id"]]:
            if action:
                def resolved(key):
                    path = f"{prefix}.{lane['id']}.{key}"
                    result = current if action[key].get("equal_previous") else values[path]
                    if action["kind"] == "Dose" and action[key].get("equal_previous"):
                        # Dose fields are grams; preceding pump targets are g/h.
                        # Derive the mass needed to keep that flow for this section.
                        result *= (end - begin) / 3600
                    values[path] = float(result)
                    return result
                finish = resolved("end")
                if action["kind"] == "Dose":
                    if finish < 0:
                        raise ValueError(f"{name}: a mass addition cannot inherit a negative pump flow.")
                    finish *= 3600 / (end - begin)
                start = resolved("start") if action["kind"] == "Ramp" else finish
            else:
                start = finish = current if lane["role"] == "T" or lane["role"].startswith("stirring") else 0.
            if lane["role"] == "valve_state" and (start not in (0, 1) or finish not in (0, 1)):
                raise ValueError(f"{name}: valve targets, including inherited values, must be 0 or 1.")
            if ROLES[lane["role"]].unit == "g/h":
                limit = 60 * (system.AS_add_max if lane["role"] == "AS" else system.sol_add_max)
                if max(start, finish) > limit + 1e-10:
                    check_limit(f"{name}: pump rate {max(start, finish) / 60:g} g/min exceeds the System settings limit {limit / 60:g} g/min.")
            if start != current and begin:
                before = begin - min(1e-6, (begin - x[-2]) / 2) if len(x) > 1 else np.nextafter(begin, 0.)
                if len(x) > 1 and before <= x[-2]:
                    raise ValueError("A stage is too short to represent its target transition.")
                x[-1] = before
                x.append(begin)
                y.append(start)
            else:
                y[-1] = start
            if lane["role"] == "T" and action:
                rate = 60 * abs(finish - start) / (end - begin)
                limit = system.v_heat_max if finish > start else system.v_cool_max
                if rate > limit + 1e-10:
                    unit = recipe.get("time_unit", "min")
                    scale = TIME_UNITS[unit]
                    minimum = 60 * abs(finish - start) / limit if limit > 0 else float("inf")
                    check_limit(f"{name}: {start:g} → {finish:g} °C in {(end - begin) / scale:g} {unit} "
                                f"requires {rate * scale / 60:g} °C/{unit}; the System settings limit is {limit * scale / 60:g} °C/{unit}. "
                                f"Minimum duration: {minimum / scale:g} {unit} ({minimum:g} s).")
                mode = action["control_mode"]
                if mode != last_mode:
                    modes.append({"time_s": float(begin), "lane": lane["id"], "mode": mode})
                    last_mode = mode
            x.append(end)
            y.append(finish)
            current = finish
        # A mass addition finishes at the specified mass, even when another
        # device has a longer profile. Other targets hold their final value.
        if schedules[lane["id"]] and schedules[lane["id"]][-1][2] and schedules[lane["id"]][-1][2]["kind"] == "Dose" and current != 0:
            x[-1] -= min(1e-6, (x[-1] - x[-2]) / 2)
            x.append(float(schedules[lane["id"]][-1][1]))
            y.append(0.)
            current = 0.
        if x[-1] < horizon:
            x.append(horizon)
            y.append(current)
        traces[lane["id"]] = (np.asarray(x), np.asarray(y))
    union = np.unique(np.concatenate([trace[0] for trace in traces.values()]))
    frame = pd.DataFrame({"t": union / 3600., **{key: np.interp(union, *trace) for key, trace in traces.items()}})
    selected = definition_window(recipe_run_stages(recipe, values))
    window = list(selected) if selected else None
    return frame, modes, window


def recipe_from_frame(frame, system, configs):
    """Make the existing component profile editable without regenerating its values."""
    from experiment.doe_campaign import DEFAULT_ASSIGNMENTS

    frame = frame.drop_duplicates().groupby("t", sort=True).last().reset_index()
    frame = frame.loc[np.r_[True, np.diff(frame["t"]) > 0]]
    lanes, used = [], set()
    for role, spec in ROLES.items():
        if role not in frame or (role != "T" and not np.any(frame[role].to_numpy())):
            continue
        candidates = [key for key, config in configs.items() if config.get("type") in spec.compatible_types
                      and key not in used]
        device = DEFAULT_ASSIGNMENTS[role] if DEFAULT_ASSIGNMENTS[role] in candidates else candidates[0] if len(candidates) == 1 else ""
        if device:
            used.add(device)
        lanes.append({"id": role, "role": role, "name": spec.label, "device_id": device,
                      "initial": float(frame[role].iloc[0])})
    stages = []
    for index in range(len(frame) - 1):
        start, end = frame.iloc[index:index + 2].to_dict(orient="records")
        duration = (end["t"] - start["t"]) * 3600
        if duration <= 0:
            continue
        actions = {}
        for lane in lanes:
            role = lane["role"]
            begin, finish = start[role], end[role]
            if role == "valve_state":
                finish = begin
            actions[role] = {"kind": "Hold" if begin == finish else "Ramp", "name": ROLES[role].label,
                             "start": parameter(begin), "end": parameter(finish), "control_mode": "Process"}
            if role == "T":
                actions[role]["start"]["equal_previous"] = True
                actions[role]["end"]["equal_previous"] = begin == finish
        stages.append({"id": f"stage{index + 1}", "name": f"Stage {index + 1}", "purpose": "Wait",
                       "duration": parameter(duration), "actions": actions})
    for lane in lanes:
        # A constant component remains one editable action when importing an
        # older design, regardless of how many boundaries other devices have.
        if stages and np.all(frame[lane["id"]].to_numpy() == lane["initial"]):
            stages[0]["actions"][lane["id"]]["until_end"] = True
            for stage in stages[1:]:
                stage["actions"].pop(lane["id"], None)
    return {"version": 1, "lanes": lanes, "stages": stages}


class RecipeCreator(ExperimentCreator):
    def __init__(self, service):
        super().__init__(service.doe_sampler, service.component_gen, service.exp_par, service.profiles, service.exp_storage)
        self.recipe = deepcopy(validate_recipe(service.exp_par.timeline))
        self.line_roles = recipe_roles(self.recipe)
        self.dimensions = sum(sampled_parameter(item) for _, item in recipe_parameters(self.recipe))
        self.weights = [1] * self.dimensions

    def set_doe_settings(self, exp_par, sampless, Type="GREEDY", Fract=0):
        self.samples, self.TYPE, self.Fract = sampless, Type, Fract
        self.if_sampling = self.dimensions > 0 and sampless > 1

    def generate_experiments(self):
        parameters = list(recipe_parameters(self.recipe))
        sampled = [key for key, value in parameters if sampled_parameter(value)]
        if self.if_sampling:
            self.DoE_samples = self.doe_sampler.doe_generator(1, self.samples, self.dimensions, self.weights, self.TYPE, self.Fract)[0]
            self.DoE_sampled_scaled = self.doe_sampler.scaler(self.DoE_samples)
            points = self.DoE_samples
        else:
            points = np.full((self.samples, self.dimensions), .5)
            self.DoE_samples = self.DoE_sampled_scaled = points.copy()
        for index, point in enumerate(points):
            values = {}
            self.DoE_val = {}
            for key, item in parameters:
                fraction = point[sampled.index(key)] if key in sampled else index % 2
                fixed = item["mode"] == "Fixed" or item["mode"] == "DoE" and not self.if_sampling
                values[key] = item["value"] if fixed else item["min"] + fraction * (item["max"] - item["min"])
            frame, modes, window = compile_recipe(self.recipe, values, self.exp_par)
            for key, item in parameters:
                if item["mode"] != "Fixed" and not item.get("equal_previous"):
                    self.DoE_val[key] = [values[key]]
            name = f"exp{index + 1}"
            for field in ("data", "data_omni", "profile", "profile_omni"):
                getattr(self.exp_storage, field)[name] = frame.copy()
            self.exp_storage.DoE[name] = pd.DataFrame({key: [values[key]] for key in self.DoE_val})
            self.parameter_snapshots[name] = {"control_modes": modes, "model_window": window,
                                               "run_stages": [vars(stage) for stage in recipe_run_stages(self.recipe, values)],
                                               **{key: {"value": value} for key, value in values.items()}}
        self.actually_sampled_experiments = len(points)

    def plot_doe_experiments(self):
        from experiment.profile_plots import profile_figure
        self.plotting_figs, self.plotting_axes = [], []
        for index, frame in enumerate(self.exp_storage.data.values(), 1):
            signals = [{"key": lane["id"], "role": lane["role"], "label": lane["name"],
                        "unit": self.line_roles[lane["id"]].unit, "times": frame["t"], "values": frame[lane["id"]]}
                       for lane in sorted(self.recipe["lanes"], key=lambda lane: lane["role"] != "T")]
            figure, axes = profile_figure(signals, index)
            self.plotting_figs.append(figure)
            self.plotting_axes.append(axes)
        return self.plotting_figs

    def export_OMNI_expts(self, loc):
        raise ValueError("Timeline recipes use assigned campaign CSV export and the native runner. Use Export DoE CSV in Settings.")
