"""Independent three-node QMOM reference for size-independent IMED benchmark kinetics.

Moments are in scaled crystal length and total batch particle number. Gaussian
quadrature closes aggregation and breakage integrals through moment order five.
"""
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.linalg import LinAlgError, eigh

from ..distribution import cell_volumes, seed_number_density


@dataclass(frozen=True)
class Reference:
    time: np.ndarray
    moments: np.ndarray
    concentration: np.ndarray
    count: np.ndarray
    aggregation_events: np.ndarray
    breakup_events: np.ndarray


def quadrature(moments):
    """Recover Gaussian nodes and positive weights from consecutive raw moments."""
    mass = moments[0]
    if mass <= 0:
        return np.empty(0), np.empty(0)
    values = moments / mass
    for size in range(len(values) // 2, 0, -1):
        indices = np.arange(size)
        hankel = values[indices[:, None] + indices]
        if np.linalg.eigvalsh(hankel)[0] <= 1e-12:
            continue
        shifted = values[indices[:, None] + indices + 1]
        nodes, vectors = eigh(shifted, hankel)
        if nodes[0] < -1e-8:
            continue
        weights = mass * (hankel[0] @ vectors)**2
        return np.maximum(nodes, 0), weights
    raise ValueError("Unrealizable QMOM moments")


def daughter_moments(length_max, shape, center_volume, deviation_volume, length_scale):
    """Moments of a Gaussian-in-volume daughter density per unit length."""
    nodes, weights = leggauss(160)
    sizes = (nodes + 1) * length_max / 2
    log_pdf = -0.5 * ((shape * sizes**3 - center_volume) / deviation_volume)**2
    weights = weights * np.exp(log_pdf - np.max(log_pdf))
    moments = weights @ ((sizes[:, None] / length_scale)**np.arange(6))
    return moments / moments[3]


def solve_reference(case, time, *, rtol=1e-9, atol=1e-11, method="DOP853"):
    """Solve the coupled QMOM and liquid balances independently of the FVM."""
    powers = np.arange(6)
    ls, ns = case.length_scale, case.number_scale
    shape, rho = case.shape, case.density
    initial_number = case.seed_number / ns
    nodes, weights = leggauss(80)
    sizes = case.seed_center + case.seed_halfwidth * nodes
    seed_pdf = np.maximum(1 - nodes**2, 0)**3
    seed_weights = weights * seed_pdf
    moments = initial_number * (seed_weights @ ((sizes[:, None] / ls)**powers)) / seed_weights.sum()
    daughters = daughter_moments(case.length_max, shape, case.daughter_center,
                                 case.daughter_width, ls)
    temperatures, feeds = case.schedules()

    def rhs(t, y):
        moments = y[:6]
        ratio, mass, tracked, _, _ = y[6:]
        temperature = np.interp(t, *temperatures)
        feed = np.interp(t, *feeds)
        anti = 1 - case.solvent_mass / mass
        liquid_volume = case.solvent_mass / case.solvent_density + (
            mass - case.solvent_mass) / case.antisolvent_density
        saturation = np.exp(case.solubility_a + case.solubility_b / (temperature + 273.15)
                            + case.solubility_d * anti)
        supersat = ratio / (1 + ratio) / saturation - 1
        positive = max(supersat, 0)
        growth = case.growth * positive
        source = case.nucleation * positive * liquid_volume / ns
        rate = growth / ls
        # Dissolution is disabled in this benchmark to isolate breakage.
        change = np.r_[source, powers[1:] * rate * moments[:-1]]
        agglomeration = disaggregation = 0.
        if positive > 0 and case.aggregation:
            x, w = quadrature(moments)
            merged = np.cbrt(x[:, None]**3 + x[None, :]**3)
            kernel = case.aggregation * ns / liquid_volume
            collisions = kernel / 2 * w[:, None] * w[None, :]
            pair_change = (merged[:, :, None]**powers
                           - x[:, None, None]**powers - x[None, :, None]**powers)
            pair_change[:, :, 3] = 0
            change += np.einsum("ij,ijk->k", collisions, pair_change)
            agglomeration = collisions.sum()
        elif positive == 0 and case.breakage and tracked > 0 and moments[0] > 0:
            removed = case.breakage * np.tanh(1) * tracked / moments[0] * moments
            delta = daughters * removed[3] - removed
            change += delta
            disaggregation = abs(delta[0])
        solute_transfer = -rho * ns * shape * ls**3 / 1e12 * (3 * rate * moments[2])
        return np.r_[change, (solute_transfer - ratio * feed) / mass, feed,
                     agglomeration - disaggregation, agglomeration, disaggregation]

    initial = np.r_[moments, case.initial_concentration / (1 - case.initial_concentration),
                    case.solvent_mass, 0, 0, 0]
    result = solve_ivp(rhs, (time[0], time[-1]), initial, t_eval=time, method=method,
                       rtol=rtol, atol=atol, max_step=case.duration / 100)
    if not result.success:
        raise RuntimeError(result.message)
    ratio, mass = result.y[6:8]
    liquid_volume = case.solvent_mass / case.solvent_density + (
        mass - case.solvent_mass) / case.antisolvent_density
    return Reference(result.t, result.y[:6], ratio / (1 + ratio),
                     result.y[0] * ns / liquid_volume, result.y[9], result.y[10])


@dataclass(frozen=True)
class CCASReference:
    time: np.ndarray
    moments: np.ndarray
    number: np.ndarray
    volume: np.ndarray
    mean_size: np.ndarray
    crystal_mass: np.ndarray
    dissolved_mass: np.ndarray
    liquid_mass: np.ndarray
    concentration: np.ndarray
    aggregation_events: np.ndarray
    breakup_events: np.ndarray


def _seed_moments(experiment, process, bounds, length_scale):
    """Match PBM seed count and mass on the selected mesh."""
    counts = seed_number_density(experiment, process, bounds) * np.diff(bounds)
    # Place each bin's count at its volume-equivalent size. This preserves
    # M0 and M3 while all six moments describe one nonnegative distribution.
    sizes = np.cbrt(cell_volumes(bounds, 1.)) / length_scale
    return counts @ (sizes[:, None]**np.arange(6))


def _ccas_daughters(br, bounds, shape, length_scale):
    if not br[0]:
        return None
    if br[6] == 1:
        # IMED's inverse-volume law diverges in count at zero length.
        # Retain the production mesh regularization for this law.
        sizes = (bounds[:-1] + bounds[1:]) / 2
        volume = shape * (bounds[:-1]**3 + bounds[1:]**3) / 2
        moments = (2 / volume) @ ((sizes[:, None] / length_scale)**np.arange(6))
    elif br[6] == 2 and br[5] > 0:
        lower, upper = bounds[[0, -1]]
        # Focus the integration knots around the Gaussian when it is narrow
        # relative to the full size domain.
        landmarks = np.cbrt((br[4] + br[5] * np.arange(-12, 13, 2)) / shape)
        edges = np.r_[lower, landmarks[(landmarks > lower) & (landmarks < upper)], upper]
        nodes, weights = leggauss(20)
        widths = np.diff(edges) / 2
        sizes = (edges[:-1, None] + widths[:, None] * (nodes + 1)).ravel()
        pdf = -0.5 * ((shape * sizes**3 - br[4]) / br[5])**2
        measure = (widths[:, None] * weights).ravel() * np.exp(pdf - np.max(pdf))
        moments = measure @ ((sizes[:, None] / length_scale)**np.arange(6))
    else:
        raise ValueError("QMOM breakage requires daughter law 1 or Gaussian law 2 with positive width")
    return moments / moments[3]


class _InvalidQMOMTrial(ValueError):
    """An adaptive integration trial left the closure's valid domain."""


def _integrate_ccas(rhs, time, initial, rtol, atol, max_step):
    # An exception from a trial stage bypasses the integrator's normal error
    # rejection. Retry with a smaller cap, keeping the caller's wall budget.
    for attempt in range(3):
        try:
            return solve_ivp(rhs, (time[0], time[-1]), initial, t_eval=time, method="DOP853",
                             rtol=rtol, atol=atol, max_step=max_step)
        except _InvalidQMOMTrial:
            if attempt == 2:
                raise
            max_step /= 4


def solve_ccas_qmom(parameters, experiment, process, solubility, settings, *, time=None,
                    rtol=1e-8, atol=1e-10, cancelled=None):
    """Run the IMED kinetics in the independent six-moment QMOM closure.

    Uses the same MAT-style dictionaries and log-parameter convention as the
    production FVM. The continuum reference assumes crystals stay on the size
    domain. It raises when resolved quadrature nodes leave that domain.
    """
    from time import monotonic

    from ..solver import (check_liquid_inventory, equilibrium_rate_scale,
                          equilibrium_relaxation_times, profile)

    pm = np.asarray(parameters, dtype=float).copy()
    if pm.shape != (35,):
        raise ValueError("Expected the 35 IMED model parameters")
    logs = [0, 3, 5, 9, 12, 16, 18, 22, 24, 28]
    pm[logs] = 10.0**pm[logs]
    if not np.isfinite(pm).all():
        raise ValueError("QMOM kinetics contain nonfinite values")
    primary, secondary, growth, dissolution, agg, br = np.split(pm, [5, 12, 18, 24, 28])
    growth_relaxation, dissolution_relaxation = equilibrium_relaxation_times(
        settings, primary, secondary, growth, dissolution)
    if agg[0] and agg[2] not in (1, 2, 3, 4, 5):
        raise ValueError("QMOM agglomeration requires one of IMED kernel codes 1–5")

    n = int(settings.get("N", 250))
    lower, upper = settings.get("Lmin", 0), settings.get("Lmax", 250)
    if n < 2 or not 0 <= lower < upper:
        raise ValueError("QMOM requires a positive size range with at least two cells")
    bounds = np.linspace(lower, upper, n + 1)
    ls = max(upper / 4, 1.)
    powers = np.arange(6)
    shape, rho = process["ShapeFactor"], process["SolidDensity"]
    solvent = process["SolvMass"]
    mass = solvent + process["AntiSolvMass"]
    seed_mass = process["SeedMass"]
    if min(shape, rho, mass, process["SolventDensity"], process["AntiSolventDensity"]) <= 0 or seed_mass < 0:
        raise ValueError("QMOM requires positive densities, shape factor, liquid mass and nonnegative seed mass")
    initial_moments = _seed_moments(experiment, process, bounds, ls)
    ns = max(initial_moments[0], 1e6)
    mass_factor = rho * ns * shape * ls**3 / 1e12
    daughter = _ccas_daughters(br, bounds, shape, ls)

    temperature = profile(experiment["TemperatureData"])
    flow_data = experiment.get("FlowRateData", [])
    flow = profile(flow_data, default=0)
    end = float(np.asarray(experiment["TemperatureData"])[0].max())
    if time is None:
        time = np.linspace(0, end, 100)
    else:
        time = np.asarray(time, dtype=float)
    if (end <= 0 or time.ndim != 1 or len(time) < 2 or time[0] != 0 or
            not np.isclose(time[-1], end) or np.any(np.diff(time) <= 0)):
        raise ValueError("QMOM output times must increase from 0 through the profile end")
    check_liquid_inventory(flow_data, flow, end, mass)
    initial_conc = (experiment["InitialConcentration"] if "InitialConcentration" in experiment
                    else np.asarray(experiment["ConcentrationData"]).reshape(2, -1)[1, 0])
    if not 0 <= initial_conc < 1:
        raise ValueError("The initial solute mass fraction must be in [0, 1)")
    a, b, c, d, e, f, g, h = (solubility[key] for key in "abcdefgh")
    deadline = monotonic() + settings.get("MaxIterationTime", 30)
    power = process["PowerRatio"]
    boundary_epsilon = 1e-10 * upper

    def velocity(law, driving_force, thermal, anti, length):
        k, exponent, size_factor, size_power, energy, modifier = law
        if k == 0:
            return np.zeros_like(length)
        # Size powers below zero are singular at a zero-length quadrature
        # abscissa. Use the smallest FVM cell center as the regularization.
        length = np.maximum(length, bounds[:2].mean()) if size_power < 0 else length
        return (k * driving_force**exponent * (1 + size_factor * length**size_power)
                * np.exp(-energy / thermal) * (1 + modifier * anti))

    def rhs(t, y):
        if cancelled is not None and cancelled():
            raise InterruptedError("QMOM comparison stopped.")
        if monotonic() > deadline:
            raise TimeoutError("QMOM exceeded its time budget")
        moments = y[:6]
        dissolved, liquid_mass, tracked, _, _ = y[6:]
        if not np.isfinite(y).all() or liquid_mass <= 0 or dissolved <= -liquid_mass:
            raise _InvalidQMOMTrial(f"Invalid QMOM liquid or moment trial state at t={t:.2f} s")
        ratio = dissolved / liquid_mass
        temperature_k = float(temperature(t)) + 273.15
        feed = float(flow(t))
        anti = 1 - solvent / liquid_mass
        liquid_volume = solvent / process["SolventDensity"] + (
            liquid_mass - solvent) / process["AntiSolventDensity"]
        csat = np.exp(a + b / temperature_k + c * np.log(temperature_k)
                      + d * anti + e * np.exp(anti) + f * anti**2
                      + g * anti / temperature_k + h * np.exp(anti) / temperature_k)
        supersat = ratio / (1 + ratio) / csat - 1
        thermal = 8.314 * temperature_k
        try:
            node, weight = quadrature(moments)
        except (ValueError, LinAlgError) as error:
            raise _InvalidQMOMTrial(f"Unrealizable QMOM trial moments at t={t:.2f} s") from error
        length = ls * node
        rate_law = growth if supersat > 0 else dissolution
        direction = 1 if supersat > 0 else -1
        driving_force = abs(supersat)
        speeds = velocity(rate_law, driving_force, thermal, anti, length)
        if len(length) and direction == 1 and length[-1] > upper + boundary_epsilon:
            volume_weights = weight * node**3
            volume_share = volume_weights[-1] / volume_weights.sum()
            raise _InvalidQMOMTrial(f"QMOM upper size boundary at t={t:.2f} s: "
                             f"trial node {length[-1]:.4g} µm exceeds Lmax={upper:g} µm "
                             f"and represents {volume_share:.2%} of quadrature crystal volume.")
        change = np.zeros(6)
        if rate_law[0]:
            k, exponent, size_factor, size_power, energy, modifier = rate_law
            if size_factor == 0 or size_power == 0:
                scalar_speed = velocity(rate_law, driving_force, thermal, anti, np.array([ls]))[0]
                change[1:] = direction * scalar_speed / ls * powers[1:] * moments[:-1]
            else:
                change[1:] = direction / ls * powers[1:] * (
                    (weight * speeds) @ (node[:, None]**powers[:-1]))
        collisions = disaggregation = 0.
        if supersat > 0:
            kp, p, ke, energy, modifier = primary
            nucleation = 0.
            if kp:
                log_s = np.log1p(supersat)
                nucleation = (kp * supersat**p * np.exp(-ke / log_s**2 - energy / thermal)
                              * (1 + modifier * anti))
            ks, s, j, mode, energy, modifier, power_exp = secondary
            if ks:
                moment = (3 * shape * ns * ls**2 * moments[2] / 1e8 if mode == 1
                          else shape * ns * ls**3 * moments[3] / 1e12)
                nucleation += (ks * supersat**s * (max(moment, 0) / liquid_volume)**j
                               * np.exp(-energy / thermal) * (1 + modifier * anti)
                               * power**power_exp)
            change += nucleation * liquid_volume / ns * (lower / ls)**powers
            if growth_relaxation is not None:
                scale = equilibrium_rate_scale(-mass_factor * change[3], dissolved,
                                               liquid_mass, csat, growth_relaxation)
                speeds *= scale
                change *= scale
            if agg[0] and len(length):
                li, lj = length[:, None], length[None, :]
                merged = np.cbrt(node[:, None]**3 + node[None, :]**3)
                if agg[2] == 1:
                    kernel = 1.
                elif agg[2] == 2:
                    floor = bounds[:2].mean()
                    kernel = (li + lj)**2 / (np.maximum(li, floor) * np.maximum(lj, floor))
                elif agg[2] == 3:
                    kernel = (li + lj)**3
                elif agg[2] == 4:
                    kernel = li**3 + lj**3
                else:
                    kernel = (li + lj)**2 * (li - lj)**2
                growth_factor = speeds**agg[1]
                speed_factor = (growth_factor[:, None] + growth_factor[None, :]) / 2
                collision_rate = (agg[0] * ns / liquid_volume * power**agg[3]
                                  * speed_factor * kernel * weight[:, None] * weight[None, :] / 2)
                collision_rate = np.where(merged * ls <= upper, collision_rate, 0)
                delta = (merged[:, :, None]**powers - node[:, None, None]**powers
                         - node[None, :, None]**powers)
                delta[:, :, 3] = 0
                change += np.einsum("ij,ijk->k", collision_rate, delta)
                collisions = collision_rate.sum()
        else:
            if dissolution_relaxation is not None:
                scale = equilibrium_rate_scale(-mass_factor * change[3], dissolved,
                                               liquid_mass, csat, dissolution_relaxation)
                speeds *= scale
                change *= scale
            if len(length) and speeds[0] > 0 and length[0] <= lower + boundary_epsilon:
                raise _InvalidQMOMTrial(f"QMOM lower size boundary at t={t:.2f} s: "
                                 f"dissolving trial node {length[0]:.4g} µm reached Lmin={lower:g} µm.")
            if daughter is not None and tracked > 0 and moments[0] > 0 and len(length):
                volume = shape * length**3
                if br[1] < 0:
                    volume = np.maximum(volume, shape * bounds[:2].mean()**3)
                frequency = br[0] * volume**br[1] * np.tanh(br[2] * (length - br[3]) + 1)
                parents = (weight * np.maximum(frequency, 0) * tracked / moments[0]) @ (
                    node[:, None]**powers)
                delta = daughter * parents[3] - parents
                change += delta
                disaggregation = abs(delta[0])
        return np.r_[change, -mass_factor * change[3],
                     feed, collisions - disaggregation, collisions, disaggregation]

    initial = np.r_[initial_moments / ns, mass * initial_conc / (1 - initial_conc), mass, 0, 0, 0]
    max_step = min(end / 100, settings.get("MaxStep", np.inf))
    relaxation = growth_relaxation if growth_relaxation is not None else dissolution_relaxation
    if relaxation is not None:
        # Keep explicit trial stages within the equilibrium relaxation scale.
        max_step = min(max_step, 4 * relaxation)
    result = _integrate_ccas(rhs, time, initial, rtol, atol, max_step)
    if not result.success or not np.isfinite(result.y).all():
        raise RuntimeError(f"QMOM integration failed. {result.message}")
    moments = result.y[:6] * ns * (ls**powers)[:, None]
    number = moments[0]
    mean = np.divide(moments[1], number, out=np.full_like(number, np.nan), where=number > 0)
    volume = shape * moments[3]
    dissolved, liquid_mass = result.y[6:8]
    return CCASReference(result.t, moments, number, volume, mean, rho * volume / 1e12,
                         dissolved, liquid_mass, dissolved / (dissolved + liquid_mass),
                         result.y[9] * ns, result.y[10] * ns)
