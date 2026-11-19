"""Named run phases and the continuous period used by kinetic models and BO."""
from dataclasses import dataclass, replace
import math


PURPOSES = ("Preparing for experiment", "Charging", "Dissolution", "Seeding", "Experiment",
            "Equilibration", "Cleaning", "Rinse", "Discharge", "Wait")
EXPERIMENT_PURPOSES = {"Experiment", "Equilibration"}


@dataclass(frozen=True)
class RunStage:
    id: str
    name: str
    purpose: str
    start_s: float
    end_s: float

    def __post_init__(self):
        if (not isinstance(self.id, str) or not self.id or not str(self.name).strip() or self.purpose not in PURPOSES
                or not all(math.isfinite(v) for v in (self.start_s, self.end_s))
                or not 0 <= self.start_s < self.end_s):
            raise ValueError("Each run stage needs a name, a purpose and a positive start/end interval in seconds.")


def stages_from_payload(payload, duration=None):
    stages = tuple(item if isinstance(item, RunStage) else RunStage(**item) for item in payload)
    if len({stage.id for stage in stages}) != len(stages):
        raise ValueError("Run stage IDs must be distinct.")
    if any(b.start_s < a.end_s - 1e-7 for a, b in zip(stages, stages[1:])):
        raise ValueError("Run stages must be ordered without overlapping intervals.")
    if duration is not None and any(stage.end_s > duration + 1e-7 for stage in stages):
        raise ValueError("Run stages must lie within the experiment's profile duration.")
    return stages


def stage_window(stages, ids=None):
    stages = stages_from_payload(stages)
    selected = [stage for stage in stages if stage.purpose in EXPERIMENT_PURPOSES] if ids is None else [stage for stage in stages if stage.id in ids]
    if ids is not None and set(ids) - {stage.id for stage in stages}:
        raise ValueError("The selected experiment-period stages are missing. Reopen Automatic loop setup and select the current stages.")
    if not selected:
        if ids is None:
            return None
        raise ValueError("Select at least one stage for the modeling experiment period.")
    if any(not math.isclose(a.end_s, b.start_s, abs_tol=1e-7, rel_tol=1e-10) for a, b in zip(selected, selected[1:])):
        raise ValueError("Select one continuous experiment period. Include the stages between the selected start and end; preparation and cleanup can stay outside it.")
    return (float(selected[0].start_s), float(selected[-1].end_s))


def definition_window(stages):
    """Leave ambiguous default labels for explicit loop selection, without blocking a draft."""
    stages = stages_from_payload(stages)
    selected = [stage for stage in stages if stage.purpose in EXPERIMENT_PURPOSES]
    if any(not math.isclose(a.end_s, b.start_s, abs_tol=1e-7, rel_tol=1e-10) for a, b in zip(selected, selected[1:])):
        return None
    return stage_window(stages)


def recipe_run_stages(recipe, values):
    elapsed, stages = 0., []
    for stage in recipe["stages"]:
        end = elapsed + values[f"{stage['id']}.duration"]
        stages.append(RunStage(stage["id"], stage["name"], stage["purpose"], elapsed, end))
        elapsed = end
    return tuple(stages)


def modeling_campaign(campaign, setup):
    if campaign is None:
        return None
    period = setup.get("experiment_period", {"mode": "definition", "stage_ids": []})
    experiments = []
    for experiment in campaign.experiments:
        if period["mode"] == "definition":
            if not experiment.run_stages:
                experiments.append(experiment)
                continue
            window = experiment.model_window or stage_window(experiment.run_stages)
            if window is None:
                raise ValueError(f"{experiment.name}: select the modeling experiment period in Automatic loop setup, or label Experiment/Equilibration stages in the definition.")
        elif period["mode"] == "stages":
            if not experiment.run_stages:
                raise ValueError(f"{experiment.name}: define named run stages before selecting a stage-based experiment period.")
            window = stage_window(experiment.run_stages, period["stage_ids"])
        else:
            duration = max(profile.points[-1].time_s for profile in experiment.profiles)
            window = (setup["recording"]["start_s"], setup["recording"]["end_s"] or duration)
        if experiment.imed_simulation is not None:
            simulation = experiment.imed_simulation
            expected = (simulation["start_s"], simulation["end_s"])
            if any(not math.isclose(a, b, abs_tol=1e-7) for a, b in zip(window, expected)):
                raise ValueError("The PBM validation simulator has a predefined experiment period. Select its crystallization window or use the definition period.")
        experiments.append(replace(experiment, model_window=window))
    return replace(campaign, experiments=tuple(experiments))


def frame_metadata(frame):
    return {key: frame.attrs[key] for key in ("run_stages", "model_window", "initial_mass_g") if key in frame.attrs}


def apply_frame_metadata(frame, metadata):
    frame.attrs.update({key: metadata[key] for key in ("run_stages", "model_window", "initial_mass_g") if key in metadata})
    return frame
