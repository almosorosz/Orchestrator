"""Headless IMED parameter-estimation forward model."""
from pathlib import Path
from typing import BinaryIO
from time import monotonic

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.io import loadmat

from .distribution import cell_volumes, seed_number_density


PBM_MODES = {
    "conservative": "SciPy PBM",
    "jax": "JAX PBM",
    "matlab_pe": "Archived MATLAB PE",
}


def _load_jax_solver():
    try:
        from . import jax_solver
    except ImportError as error:
        raise RuntimeError("JAX PBM requires JAX and Diffrax in the Python environment "
                           "running IMED. For GPU use on Windows, run IMED from source in WSL2.") from error
    return jax_solver


def pbm_backend_label(settings):
    mode = settings.get("Numerics", "conservative")
    if mode == "jax":
        device = _load_jax_solver().resolve_device(settings.get("JaxDevice", "auto"))
        return f"JAX PBM ({device.device_kind})"
    return PBM_MODES[mode]


def read_model(path: str | Path | BinaryIO, *, allow_settings_only=False):
    keys = ("ExpData", "ProcInf", "Solubility", "ModelParameter", "pbmsetting",
            "MDIteration", "MDsimulation", "PPsimulation", "summaryTable",
            "objweightfactor", "pyfitsettings", "PythonWorkflow")
    try:
        data = loadmat(path, variable_names=keys, simplify_cells=True)
    except NotImplementedError as error:
        raise ValueError("Export the IMED case from MATLAB with -v7 to import it here.") from error
    if not allow_settings_only and not any(key in data for key in keys[:4]):
        raise ValueError("The MAT file does not contain an IMED model or experiment data")
    for key in ("ExpData", "ProcInf", "ModelParameter"):
        value = data.get(key)
        if isinstance(value, dict):
            data[key] = [value]
        elif isinstance(value, np.ndarray) and not value.size:
            data[key] = []
    if isinstance(data.get("Solubility"), np.ndarray) and not data["Solubility"].size:
        data.pop("Solubility")
    if "PythonWorkflow" in data:
        from .session import history

        history(data)
    return data


def load_case(path: str | Path):
    data = read_model(path)
    for key in ("ExpData", "ProcInf", "Solubility", "ModelParameter"):
        if key not in data:
            raise ValueError(f"Missing IMED variable {key}")
    if len(data["ProcInf"]) < len(data["ExpData"]):
        raise ValueError("The MAT file needs process information for every experiment.")
    return data


def model_parameters(model):
    table = np.asarray(model["paraset"], dtype=float)
    parameters = table[:, 1].copy()
    fitted = np.asarray(model.get("para", [])).reshape(-1)
    if fitted.size:
        parameters[table[:, 0] == 1] = fitted
    return parameters


def profile(data, *, default=None):
    data = np.asarray(data, dtype=float)
    if not data.size and default is not None:
        return lambda _: default
    if data.shape == (2,):
        return lambda _: data[1]
    return interp1d(*data, fill_value="extrapolate", assume_sorted=True)


def check_liquid_inventory(flow_data, flow, end, initial_mass):
    """Reject profiles that empty the vessel, including between flow knots."""
    data = np.asarray(flow_data)
    if not data.size:
        return
    if data.shape == (2,):
        time = np.array([0, end])
        rates = np.full(2, flow(0))
    else:
        time = np.unique(np.r_[0, data[0, (data[0] > 0) & (data[0] < end)], end])
        rates = flow(time)
    if not np.isfinite(rates).all():
        raise ValueError("The antisolvent flow profile contains nonfinite values")
    width = np.diff(time)
    mass = initial_mass + np.r_[0, np.cumsum((rates[:-1] + rates[1:]) * width / 2)]
    minimum_inside = (rates[:-1] < 0) & (rates[1:] > 0)
    crossing_width = width[minimum_inside] * (-rates[:-1][minimum_inside]) / np.diff(rates)[minimum_inside]
    minimum = mass[:-1][minimum_inside] + rates[:-1][minimum_inside] * crossing_width / 2
    if (mass <= 0).any() or (minimum <= 0).any():
        raise ValueError("The antisolvent flow profile drains the liquid inventory. "
                         "Check the flow units and exclude pump return segments.")


def transport(density, speed, width, incoming=0, *, return_outflow=False, matlab_pe=False):
    """Compute the right-going high-resolution finite-volume flux."""
    epsilon = 1e-10
    downstream = np.diff(density, append=0)
    upstream = np.diff(density, prepend=0)
    center_distance = width + np.r_[width[1:], 0]
    upstream_distance = width + np.r_[0, width[:-1]]
    numerator = (upstream + epsilon) * center_distance
    denominator = (downstream + epsilon) * upstream_distance
    limiter = np.zeros_like(density)
    same_sign = (numerator > 0) == (denominator > 0)
    active = same_sign & ((numerator != 0) | (denominator != 0))
    magnitude = np.abs(numerator[active])
    limiter[active] = 2 * magnitude / (magnitude + np.abs(denominator[active]))
    face_density = density + downstream * limiter * width / center_distance
    # The outflow boundary has no upstream supply from outside the domain.
    if not matlab_pe:
        face_density[-1] = max(face_density[-1], 0)
    flux = speed * face_density
    derivative = -np.diff(flux, prepend=incoming) / width
    return (derivative, flux[-1]) if return_outflow else derivative


def aggregation_grid(volume):
    """Precompute adjacent pivots that conserve the number and volume of births."""
    i, j = np.tril_indices(len(volume))
    merged = volume[i] + volume[j]
    inside = merged <= volume[-1]
    i, j, merged = i[inside], j[inside], merged[inside]
    upper = np.searchsorted(volume, merged, side="left")
    lower = upper - 1
    upper_weight = (merged - volume[lower]) / (volume[upper] - volume[lower])
    return i, j, lower, upper, upper_weight


def aggregation_update(number, rate, pairs):
    i, j, lower, upper, upper_weight = pairs
    events = rate * number[i] * number[j] * np.where(i == j, 0.5, 1)
    n = len(number)
    delta = np.bincount(lower, events * (1 - upper_weight), minlength=n)
    delta += np.bincount(upper, events * upper_weight, minlength=n)
    delta -= np.bincount(i, events, minlength=n)
    delta -= np.bincount(j, events, minlength=n)
    return delta, events.sum()


def matlab_pe_aggregation_grid(volume, edges):
    """Reproduce the single-cell birth/death coefficients in the MATLAB PE fit."""
    i, j = np.tril_indices(len(volume))
    merged = volume[i] + volume[j]
    inside = merged < edges[-1]
    i, j, merged = i[inside], j[inside], merged[inside]
    target = np.searchsorted(edges, merged, side="right") - 1
    correction = 2 * volume[target] - merged
    birth = merged / correction * np.where(i == j, 0.5, 1)
    death = volume[target] / correction
    return i, j, target, birth, death


def matlab_pe_aggregation_update(number, rate, pairs):
    i, j, target, birth, death = pairs
    events = rate * number[i] * number[j]
    n = len(number)
    removed = events * death
    delta = np.bincount(target, events * birth, minlength=n)
    delta -= np.bincount(i, removed, minlength=n)
    delta -= np.bincount(j, removed, minlength=n)
    return delta, abs(delta.sum())


def equilibrium_relaxation_times(settings, primary, secondary, growth, dissolution):
    """Select the equilibrium limiter for phase laws that jump at saturation."""
    relaxation_time = float(settings.get("EquilibriumRelaxationTime", 1.))
    if not np.isfinite(relaxation_time) or relaxation_time <= 0:
        raise ValueError("EquilibriumRelaxationTime must be positive and finite (seconds)")
    limit_growth = ((growth[0] > 0 and growth[1] == 0)
                    or (primary[0] > 0 and primary[1] == 0 and primary[2] == 0)
                    or (secondary[0] > 0 and secondary[1] == 0))
    limit_dissolution = dissolution[0] > 0 and dissolution[1] == 0
    return (relaxation_time if limit_growth else None,
            relaxation_time if limit_dissolution else None)


def equilibrium_rate_scale(dissolved_rate, dissolved, carrier, saturation, relaxation_time):
    """Bound phase transfer by the available API surplus or deficit per second."""
    if not 0 <= saturation < 1 or dissolved_rate == 0:
        return 1.
    equilibrium_mass = carrier * saturation / (1 - saturation)
    available = equilibrium_mass - dissolved if dissolved_rate > 0 else dissolved - equilibrium_mass
    return min(1., max(available, 0.) / (abs(dissolved_rate) * relaxation_time))


def simulate(parameters, experiment, process, solubility, settings=None, *,
             return_history=False, cancelled=None):
    """Return count, concentration, and final CSD, optionally with the full ODE state.

    History row N+5 is cumulative solid API mass (g) transported beyond Lmax.
    Its particles are outside the resolved size distribution.
    Zero-order phase transfer relaxes toward equilibrium on the time scale
    settings["EquilibriumRelaxationTime"] (seconds, default 1).
    Use settings["Numerics"] = "jax" for the conservative JAX PBM, with
    settings["JaxDevice"] set to "auto" (default), "cpu", or "gpu".
    Use settings["Numerics"] = "matlab_pe" to reproduce the discretization
    used for archived MATLAB parameter fits, including its mass-balance defects.
    A cancellation callback can stop an in-progress integration.
    """
    settings = {} if settings is None else settings
    numerics = settings.get("Numerics", "conservative")
    if numerics not in PBM_MODES:
        raise ValueError(f"Unknown PBM mode {numerics!r}. Choose from {', '.join(PBM_MODES)}.")

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise InterruptedError("PBM simulation stopped.")

    check_cancelled()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if numerics == "jax":
                result = _load_jax_solver().simulate_jax(
                    parameters, experiment, process, solubility, settings,
                    return_history=return_history, cancelled=cancelled)
            else:
                result = _simulate_numpy(parameters, experiment, process, solubility, settings,
                                         numerics == "matlab_pe", return_history, check_cancelled)
            check_cancelled()
            if not all(np.isfinite(output).all() for output in result):
                raise FloatingPointError("nonfinite simulation output")
            return result
    except FloatingPointError as error:
        raise FloatingPointError(
            f"PBM numerical failure: {error}. Check the model parameter bounds and PBM grid."
        ) from error


def _simulate_numpy(parameters, experiment, process, solubility, settings,
                    matlab_pe, return_history, check_cancelled):
    n = int(settings.get("N", 250))
    bounds = np.linspace(settings.get("Lmin", 0), settings.get("Lmax", 250), n + 1)
    width = np.diff(bounds)
    if n < 2 or np.any(width <= 0) or bounds[0] < 0:
        raise ValueError("The grid requires at least two cells and increasing nonnegative bounds")
    length = bounds[1:] - width / 2
    shape = process["ShapeFactor"]
    volume = cell_volumes(bounds, shape)
    area = 3 * shape * (bounds[:-1]**2 + bounds[1:]**2) / 2
    rho = process["SolidDensity"]
    rho_solvent = process["SolventDensity"]
    rho_anti = process["AntiSolventDensity"]
    power = process["PowerRatio"]
    pm = np.array(parameters, dtype=float, copy=True)
    logs = [0, 3, 5, 9, 12, 16, 18, 22, 24, 28]
    pm[logs] = 10.0**pm[logs]
    primary, secondary, growth, dissolution, agg, br = np.split(pm, [5, 12, 18, 24, 28])
    growth_relaxation, dissolution_relaxation = (
        (None, None) if matlab_pe else
        equilibrium_relaxation_times(settings, primary, secondary, growth, dissolution)
    )
    a, b, c, d, e, f, g, h = (solubility[key] for key in "abcdefgh")
    temperature = profile(experiment["TemperatureData"])
    flow_data = experiment.get("FlowRateData", [])
    flow = profile(flow_data, default=0)
    end = np.asarray(experiment["TemperatureData"])[0].max()
    initial_conc = (experiment["InitialConcentration"] if "InitialConcentration" in experiment
                    else np.asarray(experiment["ConcentrationData"]).reshape(2, -1)[1, 0])
    mass = process["SolvMass"] + process["AntiSolvMass"]
    if end <= 0 or mass <= 0 or not 0 <= initial_conc < 1:
        raise ValueError("Expected positive duration and solvent mass and concentration in [0, 1)")
    check_liquid_inventory(flow_data, flow, end, mass)
    seed = seed_number_density(experiment, process, bounds)

    if agg[0] > 0:
        if matlab_pe:
            pairs = matlab_pe_aggregation_grid(volume, shape * bounds**3)
            aggregate = matlab_pe_aggregation_update
        else:
            pairs = aggregation_grid(volume)
            aggregate = aggregation_update
        i, j = pairs[:2]
        li, lj = length[i], length[j]
        kernels = {
            1: 1, 2: (li + lj)**2 / (li * lj), 3: (li + lj)**3,
            4: li**3 + lj**3, 5: (li + lj)**2 * (li - lj)**2,
        }
        kernel = kernels.get(agg[2], 0)

    if br[0] > 0:
        breakup = br[0] * volume**br[1] * np.tanh(br[2] * (length - br[3]) + 1)
        if br[6] == 1:
            daughters = 2 / volume
        elif br[6] == 2 and br[5] > 0:
            log_density = -0.5 * ((volume - br[4]) / br[5])**2
            daughters = np.exp(log_density - log_density.max())
        else:
            raise ValueError("Active breakage requires a supported daughter distribution")
        daughters /= np.dot(daughters, volume)

    deadline = monotonic() + settings.get("MaxIterationTime", 30)

    def rhs(t, y):
        check_cancelled()
        if monotonic() > deadline:
            raise TimeoutError("IMED forward simulation exceeded its time budget")
        if not np.isfinite(y).all():
            raise FloatingPointError("nonfinite integration state")
        density = np.maximum(y[:n], 0)
        solute, liquid_mass, solvent, agglomerated = np.maximum(y[n:n + 4], 0)
        ratio = solute if matlab_pe else solute / liquid_mass
        temperature_k = float(temperature(t)) + 273.15
        inflow = float(flow(t))
        anti = 1 - solvent
        saturation = np.exp(a + b / temperature_k + c * np.log(temperature_k)
                            + d * anti + e * np.exp(anti) + f * anti**2
                            + g * anti / temperature_k + h * np.exp(anti) / temperature_k)
        supersat = ratio / (1 + ratio) / saturation - 1
        liquid_volume = liquid_mass * (solvent / rho_solvent + anti / rho_anti)
        number = density * width
        thermal = 8.314 * temperature_k
        aggregation = disaggregation = upper_volume_flux = 0
        if supersat > 0:
            p, q, r, activation, modifier = primary
            nucleation = 0.0
            if p:
                nucleation = (p * supersat**q * np.exp(-r / np.log1p(supersat)**2)
                              * np.exp(-activation / thermal) * (1 + modifier * anti))
            p, q, r, mode, activation, modifier, power_exp = secondary
            if p:
                moment = np.dot(area if mode == 1 else volume, number)
                moment /= liquid_volume * (1e8 if mode == 1 else 1e12)
                nucleation += (p * supersat**q * moment**r * np.exp(-activation / thermal)
                               * (1 + modifier * anti) * power**power_exp)
            p, q, r, exponent, activation, modifier = growth
            speed = (p * supersat**q * (1 + r * length**exponent)
                     * np.exp(-activation / thermal) * (1 + modifier * anti))
            derivative, outflow = transport(
                density, speed, width, nucleation * liquid_volume, return_outflow=True,
                matlab_pe=matlab_pe)
            # Couple liquid consumption to the same discrete volume flux as the
            # population balance. Crystals exiting Lmax do not redissolve.
            upper_volume_flux = volume[-1] * outflow
            transfer = -np.dot(volume * width, derivative) - upper_volume_flux
            if matlab_pe:
                transfer = -np.dot(speed * area, number) - liquid_volume * nucleation * volume[0]
            elif growth_relaxation is not None:
                scale = equilibrium_rate_scale(rho * transfer / 1e12, solute, liquid_mass,
                                               saturation, growth_relaxation)
                speed *= scale
                derivative *= scale
                transfer *= scale
                upper_volume_flux *= scale
            if agg[0] > 0:
                growth_factor = speed**agg[1]
                # The archived PE driver repeats the first reactant's rate.
                pair_factor = (growth_factor[i] if matlab_pe
                               else (growth_factor[i] + growth_factor[j]) / 2)
                beta = agg[0] * pair_factor * power**agg[3] * kernel / liquid_volume
                delta, aggregation = aggregate(number, beta, pairs)
                derivative += delta / width
        else:
            p, q, r, exponent, activation, modifier = dissolution
            speed = (p * (-supersat)**q * (1 + r * length**exponent)
                     * np.exp(-activation / thermal) * (1 + modifier * anti))
            derivative = transport(density[::-1], speed[::-1], width[::-1], matlab_pe=matlab_pe)[::-1]
            transfer = -np.dot(volume * width, derivative)
            if matlab_pe:
                transfer = np.dot(speed * area, number)
            elif dissolution_relaxation is not None:
                scale = equilibrium_rate_scale(rho * transfer / 1e12, solute, liquid_mass,
                                               saturation, dissolution_relaxation)
                derivative *= scale
                transfer *= scale
            if br[0] > 0 and agglomerated > 0 and number.sum() > 0:
                removed = np.maximum(breakup * number * (agglomerated / number.sum()), 0)
                delta = daughters * np.dot(volume, removed) - removed
                derivative += delta / width
                disaggregation = abs(delta.sum())
        # Dissolved mass and the two solid inventories form a linear invariant.
        dissolved_rate = rho * transfer / 1e12
        if matlab_pe:
            dissolved_rate = -ratio * inflow / liquid_mass + dissolved_rate / liquid_mass
        rates = np.r_[derivative, dissolved_rate,
                      inflow, -solvent * inflow / liquid_mass,
                      aggregation - disaggregation, disaggregation,
                      rho * upper_volume_flux / 1e12]
        if not np.isfinite(rates).all():
            raise FloatingPointError("nonfinite PBM rate")
        return rates

    initial_ratio = initial_conc / (1 - initial_conc)
    initial_solute = initial_ratio if matlab_pe else initial_ratio * mass
    initial = np.r_[seed, initial_solute, mass,
                    process["SolvMass"] / mass, 0, 0, 0]
    time = np.linspace(0, end, 100)
    if matlab_pe:
        from .validation.matlab_ode45 import integrate_matlab_ode45

        # Old MATLAB cases used ode45 defaults, even if tolerances were saved.
        tolerances = {}
        if settings.get("UseArchivedTolerances"):
            tolerances = {"rtol": settings.get("RelTol", 1e-3),
                          "atol": settings.get("AbsTol", 1e-6)}
        state = integrate_matlab_ode45(rhs, time, initial, max_step=settings.get("MaxStep"),
                                      **tolerances)
    else:
        atol = settings.get("AbsTol", 1e-6)
        solution = solve_ivp(rhs, (0, end), initial, t_eval=time,
                             rtol=settings.get("RelTol", 1e-3), atol=atol,
                             max_step=settings.get("MaxStep", np.inf))
        if not solution.success:
            raise RuntimeError(f"IMED forward simulation failed. {solution.message}")
        state = solution.y
        # Keep the concentration ratio in the public history for existing callers.
        state[n] /= state[n + 1]
    if not np.isfinite(state).all():
        raise RuntimeError("IMED forward simulation produced nonfinite output")
    ratio, mass, solvent = state[n:n + 3]
    liquid_volume = mass * (solvent / rho_solvent + (1 - solvent) / rho_anti)
    density = state[:n]
    if not matlab_pe:
        peak = np.maximum(np.max(density, axis=0), 0.)
        roundoff = 64 * np.finfo(density.dtype).eps * peak
        # Empty bins can accumulate a few absolute tolerances of error.
        # Cap cleanup relative to the distribution so material negatives remain visible.
        density_atol = np.broadcast_to(atol, (len(initial),))[:n, None]
        tail_noise = np.minimum(8 * density_atol, 1e-8 * peak)
        tolerance = np.maximum(roundoff, tail_noise)
        density = np.where((density < 0) & (density >= -tolerance), 0., density)
    count = width @ density / liquid_volume
    outputs = (np.vstack((time, count)), np.vstack((time, ratio / (1 + ratio))),
               np.vstack((length, density[:, -1])))
    return (*outputs, state) if return_history else outputs
