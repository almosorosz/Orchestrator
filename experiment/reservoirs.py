"""Thread-safe reservoir inventory and estimated reactor mass during execution."""
from copy import deepcopy
import json
import logging
from pathlib import Path
from threading import RLock

import numpy as np
from PySide6.QtCore import QObject, Signal

from experiment.material_balance import pump_mass, delivery_times
from experiment.recording import atomic_json
from experiment.system_topology import PUMPS

logger = logging.getLogger("orchestrator.reservoirs")


class InsufficientStock(ValueError):
    pass


class ReservoirBook(QObject):
    changed = Signal()

    def __init__(self, parent=None, state_file=None):
        super().__init__(parent)
        self.lock = RLock()
        self.state_file = Path(state_file) if state_file else None
        self.units, self.configs, self.remaining, self.current_used = {}, {}, {}, {}
        self.system_mass_g = None
        self._saved = {}
        self._load_error = None
        self.active_sources = None
        if self.state_file and self.state_file.is_file():
            try:
                saved = json.loads(self.state_file.read_text(encoding="utf-8"))
                if not isinstance(saved, dict):
                    raise ValueError("Expected a mapping of refill identifiers to grams")
                if any(not isinstance(value, (float, int)) or isinstance(value, bool)
                       or not np.isfinite(value) or value < 0 for value in saved.values()):
                    raise ValueError("Saved reservoir stock must be finite, nonnegative grams")
                self._saved = saved
            except (OSError, ValueError) as error:
                self._load_error = str(error)
                logger.warning("Could not read reservoir inventory %s: %s", self.state_file, error)

    def configure(self, topology, configs, *, confirm_amounts=False):
        with self.lock:
            if confirm_amounts:
                self._saved.update({unit["inventory_id"]: unit["inventory_g"]
                                    for unit in topology.get("unit_operations", [])
                                    if unit.get("inventory_id") and unit.get("inventory_g") is not None})
                self._load_error = None
            self.units = {u["code"]: deepcopy(u) for u in topology.get("unit_operations", [])}
            self.configs = deepcopy(configs)
            self.remaining = {}
            for code, unit in self.units.items():
                amount = unit.get("inventory_g")
                if amount is None:
                    continue
                if not isinstance(amount, (float, int)) or isinstance(amount, bool) or not np.isfinite(amount) or amount < 0:
                    raise ValueError(f"{unit['name']}: reservoir amount must be nonnegative grams, or blank.")
                key = unit.get("inventory_id")
                remaining = 0. if self._load_error else (self._saved.get(key, amount) if key else amount)
                if (not isinstance(remaining, (float, int)) or isinstance(remaining, bool)
                        or not np.isfinite(remaining) or remaining < 0):
                    raise ValueError(f"{unit['name']}: saved reservoir stock must be finite, nonnegative grams. "
                                     "Enter the actual amount in Reservoir amounts before starting.")
                self.remaining[code] = float(remaining)
            self.active_sources = None
        self.changed.emit()

    def rows(self):
        with self.lock:
            return [{"code": key, "name": self.units[key]["name"], "remaining_g": amount}
                    for key, amount in self.remaining.items()
                    if self.active_sources is None or key in self.active_sources]

    def topology(self, definition):
        result = deepcopy(definition)
        with self.lock:
            for unit in result.get("unit_operations", []):
                if unit["code"] in self.remaining:
                    unit["inventory_g"] = self.remaining[unit["code"]]
        return result

    def save(self):
        with self.lock:
            if self._load_error:
                return  # Preserve the unreadable file until actual amounts have been confirmed.
            for code, amount in self.remaining.items():
                key = self.units[code].get("inventory_id")
                if key:
                    self._saved[key] = amount
            if self.state_file:
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(self.state_file, self._saved)

    def _delivery(self, experiment, elapsed):
        amounts, reactor_change, known = {}, 0., True
        reactors = {key for key, unit in self.units.items() if unit.get("material_role") == "reactor"}
        if not reactors:
            reactors = {cfg.get("connectivity", {}).get("unit") for cfg in self.configs.values() if cfg.get("type") == "huber"}
            reactors.discard(None)
            reactors.discard("")
        for profile in experiment.profiles:
            cfg = self.configs.get(profile.device_id, {})
            if cfg.get("type") not in PUMPS:
                continue
            p = cfg.get("connectivity", {})
            source, target = p.get("source"), p.get("target")
            if not source or not target or p.get("role") == "excluded":
                if any(point.target for point in profile.points):
                    known = False
                continue
            if not any(point.target for point in profile.points):
                continue
            try:
                density = self.units.get(source, {}).get("density_g_ml", cfg.get("reservoir_density_g_ml"))
                mass = pump_mass(profile, cfg, density=density, end=elapsed)
            except ValueError:
                if source in self.remaining:
                    raise
                known = False
                continue
            amounts[source] = amounts.get(source, 0.) + mass
            if target in reactors:
                reactor_change += mass
            if source in reactors:
                reactor_change -= mass
        return amounts, reactor_change, known

    def check(self, experiment, *, end_hold_s=0.):
        with self.lock:
            if self._load_error and self.remaining:
                raise ValueError("The saved reservoir inventory could not be read. Open Reservoir amounts, "
                                 "enter the actual container amounts and save before starting.")
            end = max(p.points[-1].time_s for p in experiment.profiles) + end_hold_s
            times = sorted({point.time_s for profile in experiment.profiles for point in profile.points} | {end})
            times = sorted(set(times) | {float(time) for profile in experiment.profiles
                           if self.configs.get(profile.device_id, {}).get("type") in PUMPS
                           for time in delivery_times(profile, self.configs[profile.device_id], end=end)})
            required = {}
            for time in times:
                amounts, _, _ = self._delivery(experiment, time)
                for source, mass in amounts.items():
                    required[source] = max(required.get(source, 0.), mass)
            for left, right in zip(times, times[1:]):
                middle = left + (right - left) / 2
                if not left < middle < right:
                    continue
                a, _, _ = self._delivery(experiment, left)
                m, _, _ = self._delivery(experiment, middle)
                b, _, _ = self._delivery(experiment, right)
                for source in set(a) | set(m) | set(b):
                    # Piecewise-linear flow gives quadratic cumulative delivery.
                    qa, qm, qb = a.get(source, 0.), m.get(source, 0.), b.get(source, 0.)
                    curvature = 2 * (qb + qa - 2 * qm)
                    slope = qb - qa - curvature
                    if curvature < 0:
                        fraction = -slope / (2 * curvature)
                        if 0 < fraction < 1:
                            required[source] = max(required.get(source, 0.), qa + slope * fraction + curvature * fraction**2)
            for source, mass in required.items():
                if source in self.remaining and mass > self.remaining[source] + 1e-7:
                    raise InsufficientStock(f"Insufficient reservoir stock for {experiment.name}: {self.units[source]['name']} "
                                            f"has {self.remaining[source]:g} g; the next experiment requires {mass:g} g. "
                                            "Refill it and update Reservoir amounts in the Control Panel before restarting.")
            return required

    def begin(self, experiment, *, end_hold_s=0.):
        self.check(experiment, end_hold_s=end_hold_s)
        with self.lock:
            self.current_used = {}
            self._scale_start = {}
            self._last_reactor_change = 0.
            self._last_valve = False
            explicit = [unit.get("initial_system_mass_g") for unit in self.units.values()
                        if unit.get("initial_system_mass_g") is not None]
            initial = sum(explicit) if explicit else experiment.initial_mass_g
            if initial is None:
                initial = sum(unit.get("initial_system_mass_g", 0.) or 0. for unit in self.units.values()
                              if unit.get("material_role") == "reactor")
            self.system_mass_g = float(initial)
            self.active_sources = {source for source, mass in self.check(experiment, end_hold_s=end_hold_s).items() if mass > 0}
        self.changed.emit()

    def advance(self, experiment, elapsed, telemetry):
        with self.lock:
            used, reactor_change, known = self._delivery(experiment, elapsed)
            # A reservoir scale measures the total draw through all its pumps.
            for source in used:
                scales = [device for device, cfg in self.configs.items() if cfg.get("type") == "mettler"
                          and cfg.get("connectivity", {}).get("unit") == source]
                if len(scales) == 1:
                    weight = telemetry.get(scales[0], {}).get("weight_g")
                    if weight is not None and np.isfinite(weight):
                        self._scale_start.setdefault(source, float(weight) + used[source])
                        measured = self._scale_start[source] - float(weight)
                        used[source] = max(0., measured) if used[source] >= 0 else measured
            for source, cumulative in used.items():
                delta = cumulative - self.current_used.get(source, 0.)
                if source in self.remaining:
                    self.remaining[source] = max(0., self.remaining[source] - delta)
                self.current_used[source] = cumulative
            # Source-scale differences also correct the reactor addition estimate.
            planned, _, _ = self._delivery(experiment, elapsed)
            for source, cumulative in used.items():
                targets = {cfg.get("connectivity", {}).get("target") for cfg in self.configs.values()
                           if cfg.get("type") in PUMPS and cfg.get("connectivity", {}).get("source") == source}
                reactor_units = {key for key, unit in self.units.items() if unit.get("material_role") == "reactor"}
                if not reactor_units:
                    reactor_units = {cfg.get("connectivity", {}).get("unit") for cfg in self.configs.values() if cfg.get("type") == "huber"}
                if len(targets) == 1 and targets <= reactor_units:
                    reactor_change += cumulative - planned[source]
            opened = any(telemetry.get(key, {}).get("target_state", telemetry.get(key, {}).get("target", 0))
                         for key, cfg in self.configs.items() if cfg.get("type") == "nidaq_valve")
            if known and self.system_mass_g is not None:
                self.system_mass_g = max(0., self.system_mass_g + reactor_change - self._last_reactor_change)
                if opened and not self._last_valve:
                    self.system_mass_g = 0.
            else:
                self.system_mass_g = None
            self._last_reactor_change, self._last_valve = reactor_change, bool(opened)
        self.changed.emit()
        return self.system_mass_g
