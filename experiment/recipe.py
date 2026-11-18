"""Editable, device-lane recipes compiled into the shared campaign profiles."""
from copy import deepcopy
from math import isfinite

import numpy as np
import pandas as pd

from experiment.doe.doe_creator import ExperimentCreator
from experiment.doe_campaign import ROLES, ProfileRole


PURPOSES = ("Charging", "Experiment", "Equilibration", "Cleaning", "Rinse", "Discharge", "Wait")
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


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or recipe.get("version") != 1:
        raise ValueError("Choose an Experiment Definition recipe.")
    lanes, stages = recipe["lanes"], recipe["stages"]
    for entries, label in ((lanes, "device lines"), (stages, "stages")):
        if not isinstance(entries, list) or not entries or len({item["id"] for item in entries}) != len(entries):
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
    for stage in stages:
        if stage["purpose"] not in PURPOSES or set(stage["actions"]) - by_id.keys():
            raise ValueError("Each stage needs a valid purpose and existing device lines.")
        for lane_id, action in stage["actions"].items():
            role = by_id[lane_id]["role"]
            if action["kind"] not in {"Hold", "Ramp", "Dose"} or (role == "valve_state" and action["kind"] != "Hold"):
                raise ValueError("Valve actions must hold a binary state. Other devices support holds and ramps.")
            if action["kind"] == "Dose" and ROLES[role].unit != "g/h":
                raise ValueError("Mass additions need a solution, solvent, antisolvent or cleaning pump line.")
            if action["kind"] == "Dose":
                amount = action["end"]
                if (amount["value"] if amount["mode"] == "Fixed" else amount["min"]) < 0:
                    raise ValueError("A mass addition must be nonnegative.")
            if role == "T" and action.get("control_mode") not in {"Jacket", "Process"}:
                raise ValueError("Choose jacket or internal vessel (Process) temperature control.")
    for name, item in recipe_parameters(recipe):
        if item["mode"] not in {"Fixed", "DoE", "Alternate"} or any(
            isinstance(item[key], bool) or not isinstance(item[key], (int, float)) or not isfinite(item[key])
            for key in ("value", "min", "max")
        ) or item["min"] > item["max"]:
            raise ValueError(f"Check the value and bounds for {name}.")
        low = item["value"] if item["mode"] == "Fixed" else item["min"]
        high = item["value"] if item["mode"] == "Fixed" else item["max"]
        if name.endswith(".duration") and low <= 0:
            raise ValueError("Stage durations must be positive in every experiment.")
        lane_id = name.split(".")[1] if not name.endswith(".duration") else None
        if lane_id and by_id[lane_id]["role"] == "valve_state" and (low not in (0, 1) or high not in (0, 1)
                                                                 or item["mode"] == "DoE"):
            raise ValueError("Valve states must be fixed at 0/1 or alternate between 0 and 1.")
    return recipe


def compile_recipe(recipe, values, system):
    lanes = recipe["lanes"]
    times = np.r_[0., np.cumsum([values[f"{stage['id']}.duration"] for stage in recipe["stages"]])]
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Stage durations must produce an increasing finite timeline.")
    traces, modes = {}, []
    for lane in lanes:
        current = lane["initial"]
        x, y = [0.], [current]
        last_mode = None
        for index, stage in enumerate(recipe["stages"]):
            action = stage["actions"].get(lane["id"])
            begin, end = times[index:index + 2]
            if action:
                finish = values[f"{stage['id']}.{lane['id']}.end"]
                if action["kind"] == "Dose":
                    finish *= 3600 / (end - begin)
                start = values[f"{stage['id']}.{lane['id']}.start"] if action["kind"] == "Ramp" else finish
            else:
                start = finish = current if lane["role"] == "T" or lane["role"].startswith("stirring") else 0.
            if ROLES[lane["role"]].unit == "g/h":
                limit = 60 * (system.AS_add_max if lane["role"] == "AS" else system.sol_add_max)
                if max(start, finish) > limit + 1e-10:
                    raise ValueError(f"{stage['name']} exceeds the System Settings pump addition limit.")
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
                    raise ValueError(f"{stage['name']} exceeds the System Settings temperature ramp limit.")
                mode = action["control_mode"]
                if mode != last_mode:
                    modes.append({"time_s": float(begin), "lane": lane["id"], "mode": mode})
                    last_mode = mode
            x.append(end)
            y.append(finish)
            current = finish
        if lane["role"] in {"AS", "solution", "solvent", "clean_sol", "seed", "valve_state"} and y[-1] != 0:
            x.append(float(times[-1] + 1e-6))
            y.append(0.)
        traces[lane["id"]] = (np.asarray(x), np.asarray(y))
    union = np.unique(np.concatenate([trace[0] for trace in traces.values()]))
    frame = pd.DataFrame({"t": union / 3600., **{key: np.interp(union, *trace) for key, trace in traces.items()}})
    active = [i for i, stage in enumerate(recipe["stages"]) if stage["purpose"] in {"Experiment", "Equilibration"}]
    if active and active != list(range(min(active), max(active) + 1)):
        raise ValueError("Keep Experiment and Equilibration stages together so modeling uses one continuous window.")
    window = [float(times[min(active)]), float(times[max(active) + 1])] if active else None
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
        stages.append({"id": f"stage{index + 1}", "name": f"Stage {index + 1}", "purpose": "Wait",
                       "duration": parameter(duration), "actions": actions})
    return {"version": 1, "lanes": lanes, "stages": stages}


class RecipeCreator(ExperimentCreator):
    def __init__(self, service):
        super().__init__(service.doe_sampler, service.component_gen, service.exp_par, service.profiles, service.exp_storage)
        self.recipe = deepcopy(validate_recipe(service.exp_par.timeline))
        self.line_roles = recipe_roles(self.recipe)
        self.dimensions = sum(item["mode"] == "DoE" and item["min"] < item["max"] for _, item in recipe_parameters(self.recipe))
        self.weights = [1] * self.dimensions

    def set_doe_settings(self, exp_par, sampless, Type="GREEDY", Fract=0):
        self.samples, self.TYPE, self.Fract = sampless, Type, Fract
        self.if_sampling = self.dimensions > 0 and sampless > 1

    def generate_experiments(self):
        parameters = list(recipe_parameters(self.recipe))
        sampled = [key for key, value in parameters if value["mode"] == "DoE" and value["min"] < value["max"]]
        if self.if_sampling:
            self.DoE_samples = self.doe_sampler.doe_generator(1, self.samples, self.dimensions, self.weights, self.TYPE, self.Fract)[0]
            self.DoE_sampled_scaled = self.doe_sampler.scaler(self.DoE_samples)
            points = self.DoE_samples
        else:
            points = np.full((self.samples, self.dimensions), .5)
            self.DoE_samples = self.DoE_sampled_scaled = points.copy()
        for index, point in enumerate(points):
            values = {}
            for key, item in parameters:
                fraction = point[sampled.index(key)] if key in sampled else index % 2
                fixed = item["mode"] == "Fixed" or item["mode"] == "DoE" and not self.if_sampling
                values[key] = item["value"] if fixed else item["min"] + fraction * (item["max"] - item["min"])
                if item["mode"] != "Fixed":
                    self.DoE_val[key] = [values[key]]
            frame, modes, window = compile_recipe(self.recipe, values, self.exp_par)
            name = f"exp{index + 1}"
            for field in ("data", "data_omni", "profile", "profile_omni"):
                getattr(self.exp_storage, field)[name] = frame.copy()
            self.exp_storage.DoE[name] = pd.DataFrame({key: [values[key]] for key in self.DoE_val})
            self.parameter_snapshots[name] = {"control_modes": modes, "model_window": window,
                                               **{key: {"value": value} for key, value in values.items()}}
        self.actually_sampled_experiments = len(points)

    def plot_doe_experiments(self):
        from matplotlib.figure import Figure
        figures = []
        for name, frame in self.exp_storage.data.items():
            figure = Figure(figsize=(10, 2.2 * len(self.line_roles)), layout="constrained")
            for index, (key, role) in enumerate(self.line_roles.items(), 1):
                axis = figure.add_subplot(len(self.line_roles), 1, index)
                axis.plot(frame["t"] * 3600, frame[key])
                axis.set_ylabel(f"{role.label} ({role.unit})")
                axis.set_xlabel("Time (s)")
                axis.grid(alpha=.2)
            figure.suptitle(name)
            figures.append(figure)
        return figures

    def export_OMNI_expts(self, loc):
        raise ValueError("Timeline recipes use assigned campaign CSV export and the native runner. Use Export DoE CSV in Settings.")
