"""Import the recorded measurements from a one-off modeling recipe."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bo.profiles import case_runs
from experiment import loop_outputs
from experiment.adaptive_loop import read_completed_recordings
from experiment.recording import atomic_json
from imed_python.native_case import save_native_case


@dataclass(frozen=True)
class RecipeFeedback:
    mode: str
    campaign_id: str
    experiment_id: str
    recording: dict
    mapping: dict
    roles: dict
    process: dict
    window: tuple
    solubility: dict
    source: str
    seed_template: dict
    model_root: str

    def read(self, folder):
        experiments, processes, runs = read_completed_recordings(
            folder, (self.experiment_id,), self.recording, self.mapping,
            {self.experiment_id: self.process}, {self.experiment_id: self.window},
            require_initial_concentration=self.mode == "imed", roles=self.roles,
            mixing=self.process.get("MixingSettings") if self.mode == "imed" else None)
        for experiment in experiments:
            experiment.update(RecipeSource=self.source)
            if self.process["SeedMass"]:
                for key, values in self.seed_template.items():
                    if not np.size(experiment.get(key)):
                        experiment[key] = values
        return {"ExpData": experiments, "ProcInf": processes, "Solubility": self.solubility}, runs

    def merge(self, state, measured, folder):
        if self.mode == "bo":
            added = case_runs(measured, folder, controls=self.roles)
            existing = {run.get("recording") for run in state["runs"]}
            added = [run for run in added if run.get("recording") not in existing]
            for run in added:
                run["recipe_stage"] = self.source
            start = len(state["runs"])
            return {**state, "runs": [*state["runs"], *added],
                    "selected": [*state["selected"], *range(start, start + len(added))], "source": str(folder)}
        prior = state.get("ExpData", [])
        existing = {experiment.get("RecordingFile") for experiment in prior}
        pairs = [(experiment, process) for experiment, process in zip(measured["ExpData"], measured["ProcInf"])
                 if experiment.get("RecordingFile") not in existing]
        processes = [{**process, "No": index} for index, (_, process) in enumerate(pairs, len(prior) + 1)]
        return {**state, "ExpData": [*prior, *(experiment for experiment, _ in pairs)],
                "ProcInf": [*state.get("ProcInf", []), *processes],
                "Solubility": state.get("Solubility") or measured["Solubility"]}

    def save(self, folder, state, runs, report):
        root = Path(self.model_root) if self.model_root else None
        target = root / f"_recipe_{self.mode}_{self.campaign_id[:12]}" if root and root.is_dir() else folder / "model_inputs"
        target.mkdir(parents=True, exist_ok=True)
        loop_outputs.saved_recordings(target, runs, report)
        if self.mode == "imed":
            loop_outputs.imed_inputs(target, state)
            return save_native_case(target / "recipe_inputs.imed.json", state)
        loop_outputs.bo_inputs(target, state)
        path = target / "recipe_inputs.bo.json"
        atomic_json(path, state)
        return path
