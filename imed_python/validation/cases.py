"""Matched FVM/QMOM cases for cooling, antisolvent, and combined crystallization."""
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss


@dataclass(frozen=True)
class Case:
    profile: str
    mechanisms: str
    duration: float = 60.
    growth: float = 0.65
    initial_concentration: float = 0.1
    solvent_mass: float = 68.
    solvent_density: float = 0.86
    antisolvent_density: float = 1.
    density: float = 1.4
    shape: float = 1.
    seed_number: float = 2e7
    seed_center: float = 12.
    seed_halfwidth: float = 4.
    length_scale: float = 20.
    number_scale: float = 2e7
    length_max: float = 65.
    solubility_b: float = -1200.
    solubility_d: float = -0.35
    daughter_center: float = 1400.
    daughter_width: float = 360.

    @property
    def solubility_a(self):
        return np.log(0.079) - self.solubility_b / 298.15

    @property
    def nucleation(self):
        return 0. if self.mechanisms == "growth" else 1.3e4

    @property
    def aggregation(self):
        return 5e-8 if self.mechanisms == "full" else 0.

    @property
    def breakage(self):
        return 0.12 if self.mechanisms == "full" else 0.

    def schedules(self):
        time = [0., 15., 35., 45., self.duration]
        cooling = self.profile in ("cooling", "combined")
        adding = self.profile in ("antisolvent", "combined")
        temperature = [25., 12., 12., 12., 12.] if cooling else [25.] * len(time)
        # Breakage in IMED is restricted to undersaturation. Reheat cooling
        # cases near the end, and use a late dilution pulse in AS-only cases.
        if self.breakage and cooling:
            temperature[-2:] = [55., 55.]
        feed = [0., 0.55, 0.55, 0.0, 0.0] if adding else [0.] * len(time)
        if self.breakage and self.profile == "antisolvent":
            feed[-2:] = [2.8, 2.8]
        return np.array([time, temperature]), np.array([time, feed])

    def inputs(self, cells):
        temperature, feed = self.schedules()
        x = np.linspace(0, self.length_max, 2601)
        seed_pdf = np.maximum(1 - ((x - self.seed_center) / self.seed_halfwidth)**2, 0)**3
        nodes, weights = leggauss(80)
        seed_weights = weights * (1 - nodes**2)**3
        sizes = self.seed_center + self.seed_halfwidth * nodes
        seed_mass = (self.seed_number * self.density * self.shape / 1e12
                     * (seed_weights @ sizes**3) / seed_weights.sum())
        experiment = {"TemperatureData": temperature, "FlowRateData": feed,
                      "ConcentrationData": [[0], [self.initial_concentration]],
                      "InitialCSDsData": [x, seed_pdf * x**3]}
        process = {"ShapeFactor": self.shape, "SolidDensity": self.density,
                   "SolventDensity": self.solvent_density,
                   "AntiSolventDensity": self.antisolvent_density,
                   "SolvMass": self.solvent_mass, "AntiSolvMass": 0.,
                   "PowerRatio": 1., "SeedMass": seed_mass}
        solubility = dict.fromkeys("abcdefgh", 0.)
        solubility.update(a=self.solubility_a, b=self.solubility_b, d=self.solubility_d)
        params = np.zeros(35)
        params[[0, 3, 5, 9, 16, 18, 22, 24, 28]] = -np.inf
        params[1] = params[13] = 1.
        params[12] = np.log10(self.growth)
        if self.nucleation:
            params[0] = np.log10(self.nucleation)
        if self.aggregation:
            params[24] = np.log10(self.aggregation)
            params[26] = 1.
        if self.breakage:
            params[28] = np.log10(self.breakage)
            params[32:35] = self.daughter_center, self.daughter_width, 2
        settings = {"Lmin": 0., "Lmax": self.length_max, "N": cells,
                    "RelTol": 5e-7, "AbsTol": 1e-8, "MaxStep": self.duration / 100,
                    "MaxIterationTime": 120}
        return params, experiment, process, solubility, settings
