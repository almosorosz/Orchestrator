"""Conservative JAX PBM from the validated MainCode_Agg_Br notebook."""
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import ClipStepSizeController, Event, ODETerm, PIDController, RESULTS, SaveAt, Tsit5, diffeqsolve
from jax.experimental import io_callback
from scipy.interpolate import interp1d

from .distribution import cell_volumes
from .solver import check_liquid_inventory

# Match the precision used for the notebook comparisons.
jax.config.update("jax_enable_x64", True)


def resolve_device(requested):
    if requested == "auto":
        return jax.config.jax_default_device or jax.devices()[0]
    if requested not in ("cpu", "gpu"):
        raise ValueError("JaxDevice must be auto, cpu, or gpu.")
    try:
        devices = jax.devices(requested)
    except RuntimeError as error:
        if requested == "cpu":
            raise
        setup = ("Run IMED from source in WSL2 with GPU-enabled JAX."
                 if sys.platform == "win32" else
                 "Check the GPU driver and the JAX accelerator installation.")
        raise RuntimeError(f"JAX could not initialize a GPU. {setup}") from error
    if not devices:
        raise RuntimeError(f"JAX did not find a {requested.upper()} device.")
    return devices[0]


class PBMArguments(NamedTuple):
    primary: jax.Array
    secondary: jax.Array
    growth: jax.Array
    dissolution: jax.Array
    aggregation: jax.Array
    breakage: jax.Array
    length: jax.Array
    width: jax.Array
    area: jax.Array
    volume: jax.Array
    pair_left: jax.Array
    pair_right: jax.Array
    birth_lower: jax.Array
    birth_upper: jax.Array
    upper_weight: jax.Array
    pair_kernel: jax.Array
    breakup_frequency: jax.Array
    daughters: jax.Array
    temperature: jax.Array
    feed: jax.Array
    solubility: jax.Array
    properties: jax.Array
    equilibrium_relaxation_time: jax.Array


def transport(density, speed, width, incoming=0.):
    """Right-going finite-volume transport. Reverse the mesh for dissolution."""
    downstream = jnp.diff(density, append=0.)
    upstream = jnp.diff(density, prepend=0.)
    center_distance = width + jnp.concatenate((width[1:], jnp.zeros(1)))
    upstream_distance = width + jnp.concatenate((jnp.zeros(1), width[:-1]))
    numerator = (upstream + 1e-10) * center_distance
    denominator = (downstream + 1e-10) * upstream_distance
    magnitude = jnp.abs(numerator)
    divisor = magnitude + jnp.abs(denominator)
    active = ((numerator > 0) == (denominator > 0)) & (divisor > 0)
    limiter = jnp.where(active, 2 * magnitude / jnp.where(active, divisor, 1.), 0.)
    face_density = density + downstream * limiter * width / center_distance
    face_density = face_density.at[-1].set(jnp.maximum(face_density[-1], 0.))
    flux = speed * face_density
    return -jnp.diff(flux, prepend=incoming) / width, flux[-1]


def velocity(law, driving_force, length, thermal, antisolvent):
    rate, exponent, size_factor, size_power, activation, modifier = law
    return (rate * driving_force**exponent * (1 + size_factor * length**size_power)
            * jnp.exp(-activation / thermal) * (1 + modifier * antisolvent))


def HR_FVM_rhs(t, state, args):
    """Conservative JAX RHS. State row N stores dissolved API grams internally."""
    width, volume = args.width, args.volume
    n = width.size
    density = jnp.maximum(state[:n], 0.)
    dissolved, carrier_mass, solvent, agglomerated = jnp.maximum(state[n:n + 4], 0.)
    antisolvent = 1 - solvent
    thermal_temperature = jnp.interp(t, *args.temperature, left="extrapolate", right="extrapolate") + 273.15
    inflow = jnp.interp(t, *args.feed, left="extrapolate", right="extrapolate")
    rho, rho_solvent, rho_antisolvent, power = args.properties
    liquid_volume = carrier_mass * (solvent / rho_solvent + antisolvent / rho_antisolvent)
    a, b, c, d, e, f, g, h = args.solubility
    temperature = thermal_temperature
    saturation = jnp.exp(
        a + b / temperature + c * jnp.log(temperature) + d * antisolvent
        + e * jnp.exp(antisolvent) + f * antisolvent**2
        + g * antisolvent / temperature + h * jnp.exp(antisolvent) / temperature
    )
    ratio = dissolved / carrier_mass
    supersaturation = ratio / (1 + ratio) / saturation - 1
    number = density * width
    thermal = 8.314 * temperature
    # A zero-order rate jumps at saturation. Restrict the phase transfer to
    # the available excess (or deficit) API over the relaxation timescale.
    csat = jnp.minimum(saturation, 1 - jnp.finfo(state.dtype).eps)
    excess_api = dissolved - carrier_mass * csat / (1 - csat)
    rho_scale = rho / 1e12

    def equilibrium_scale(transfer, available, enabled):
        rate = jnp.abs(transfer) * rho_scale
        limit = jnp.maximum(available, 0.) / args.equilibrium_relaxation_time
        limited = enabled & (saturation < 1.) & (rate > limit)
        return jnp.where(limited, limit / jnp.where(limited, rate, 1.), 1.)

    def growing(_):
        kp, order, barrier, activation, modifier = args.primary
        primary = jax.lax.cond(
            kp != 0.,
            lambda: kp * supersaturation**order * jnp.exp(-barrier / jnp.log1p(supersaturation)**2)
            * jnp.exp(-activation / thermal) * (1 + modifier * antisolvent),
            lambda: jnp.zeros_like(supersaturation),
        )
        ks, order, exponent, mode, activation, modifier, power_exponent = args.secondary

        def secondary_rate():
            surface_based = mode == 1
            moment = jnp.dot(jnp.where(surface_based, args.area, volume), number)
            moment /= liquid_volume * jnp.where(surface_based, 1e8, 1e12)
            return (ks * supersaturation**order * moment**exponent * jnp.exp(-activation / thermal)
                    * (1 + modifier * antisolvent) * power**power_exponent)

        secondary = jax.lax.cond(ks != 0., secondary_rate, lambda: jnp.zeros_like(supersaturation))
        speed = velocity(args.growth, supersaturation, args.length, thermal, antisolvent)
        derivative, outflow = transport(density, speed, width, (primary + secondary) * liquid_volume)
        upper_volume_flux = volume[-1] * outflow
        transfer = -jnp.dot(volume * width, derivative) - upper_volume_flux
        primary_zero_order = ((args.primary[0] > 0) & (args.primary[1] == 0)
                              & (args.primary[2] == 0))
        secondary_zero_order = (args.secondary[0] > 0) & (args.secondary[1] == 0)
        growth_zero_order = (args.growth[0] > 0) & (args.growth[1] == 0)
        scale = equilibrium_scale(transfer, excess_api,
                                  primary_zero_order | secondary_zero_order | growth_zero_order)
        derivative = derivative * scale
        transfer = transfer * scale
        upper_volume_flux = upper_volume_flux * scale
        speed = speed * scale
        ka, growth_power, _, power_exponent = args.aggregation

        def collisions():
            i, j = args.pair_left, args.pair_right
            growth_factor = speed**growth_power
            pair_factor = (growth_factor[i] + growth_factor[j]) / 2
            events = (ka * power**power_exponent / liquid_volume
                      * args.pair_kernel * pair_factor * number[i] * number[j])
            delta = jnp.zeros_like(number)
            delta = delta.at[args.birth_lower].add(events * (1 - args.upper_weight))
            delta = delta.at[args.birth_upper].add(events * args.upper_weight)
            delta = delta.at[i].add(-events)
            delta = delta.at[j].add(-events)
            return delta / width, events.sum()

        addition, events = jax.lax.cond(
            ka > 0., collisions, lambda: (jnp.zeros_like(density), jnp.zeros_like(supersaturation)),
        )
        return derivative + addition, transfer, events, jnp.zeros_like(events), upper_volume_flux

    def dissolving(_):
        speed = velocity(args.dissolution, -supersaturation, args.length, thermal, antisolvent)
        reversed_derivative, _ = transport(density[::-1], speed[::-1], width[::-1])
        derivative = reversed_derivative[::-1]
        transfer = -jnp.dot(volume * width, derivative)
        dissolution_zero_order = (args.dissolution[0] > 0) & (args.dissolution[1] == 0)
        scale = equilibrium_scale(transfer, -excess_api, dissolution_zero_order)
        derivative = derivative * scale
        transfer = transfer * scale
        total_number = number.sum()

        def fragments():
            removed = jnp.maximum(args.breakup_frequency * number * (agglomerated / total_number), 0.)
            delta = args.daughters * jnp.dot(volume, removed) - removed
            return delta / width, jnp.abs(delta.sum())

        addition, broken = jax.lax.cond(
            (args.breakage[0] > 0.) & (agglomerated > 0.) & (total_number > 0.),
            fragments, lambda: (jnp.zeros_like(density), jnp.zeros_like(supersaturation)),
        )
        zero = jnp.zeros_like(broken)
        return derivative + addition, transfer, zero, broken, zero

    derivative, transfer, events, broken, upper_volume_flux = jax.lax.cond(
        supersaturation > 0., growing, dissolving, operand=None,
    )
    # Dissolved mass plus resolved and off-grid crystal masses is a linear invariant.
    return jnp.concatenate((derivative, jnp.stack((
        rho * transfer / 1e12,
        inflow,
        -solvent * inflow / carrier_mass,
        events - broken,
        broken,
        rho * upper_volume_flux / 1e12,
    ))))


def stop_requested(cancelled, t, y, args, **kwargs):
    """Poll the host's stop flag while Diffrax is running compiled steps."""
    return io_callback(cancelled, jax.ShapeDtypeStruct((), jnp.bool_))


def run_model(initial, args, time, *, rtol, atol, max_step, profile_knots, max_steps, throw=True,
              cancelled=None):
    """Run Diffrax on JAX arrays. The returned raw state stores dissolved API grams."""
    controller = PIDController(rtol=rtol, atol=atol, dtmax=max_step)
    if profile_knots.size:
        controller = ClipStepSizeController(controller, step_ts=profile_knots)
    event = Event(jax.tree_util.Partial(stop_requested, cancelled)) if cancelled is not None else None
    return diffeqsolve(
        ODETerm(HR_FVM_rhs), Tsit5(), t0=time[0], t1=time[-1], dt0=None,
        y0=initial, args=args, saveat=SaveAt(ts=time),
        stepsize_controller=controller, max_steps=max_steps, throw=throw, event=event,
    )


def profile_data(data, end, *, default=None):
    points = np.asarray(data, dtype=float)
    if not points.size and default is not None:
        points = np.array([0., default])
    if points.shape == (2,):
        points = np.array([[0., end], [points[1], points[1]]])
    if (points.ndim != 2 or points.shape[0] != 2 or points.shape[1] < 2
            or not np.isfinite(points).all() or np.any(np.diff(points[0]) <= 0)):
        raise ValueError("Profiles require at least two finite, strictly increasing knots.")
    return points


def prepare_collisions(length, volume, kernel_code):
    i, j = np.tril_indices(length.size)
    merged = volume[i] + volume[j]
    upper = np.clip(np.searchsorted(volume, merged, side="left"), 1, length.size - 1)
    lower = upper - 1
    weight = np.clip((merged - volume[lower]) / (volume[upper] - volume[lower]), 0, 1)
    li, lj = length[i], length[j]
    kernels = {
        1: lambda: 1., 2: lambda: (li + lj)**2 / (li * lj),
        3: lambda: (li + lj)**3, 4: lambda: li**3 + lj**3,
        5: lambda: (li + lj)**2 * (li - lj)**2,
    }
    kernel = kernels.get(kernel_code, lambda: 0.)()
    # Keep a fixed shape for JIT reuse. Mask excluded pairs before evaluating rates.
    pair_kernel = np.where(merged <= volume[-1], kernel * np.where(i == j, 0.5, 1.), 0.)
    return i, j, lower, upper, weight, pair_kernel


def prepare_jax_case(parameters, experiment, process, solubility, settings):
    n = int(settings.get("N", 250))
    lower, upper = settings.get("Lmin", 0.), settings.get("Lmax", 250.)
    if n < 2 or not 0 <= lower < upper or not np.isfinite(upper):
        raise ValueError("Use at least two bins and increasing nonnegative size bounds.")
    edges = np.linspace(lower, upper, n + 1)
    width = np.diff(edges)
    length = edges[1:] - width / 2
    shape = process["ShapeFactor"]
    volume = cell_volumes(edges, shape)
    area = 3 * shape * (edges[:-1]**2 + edges[1:]**2) / 2
    properties = np.array([process["SolidDensity"], process["SolventDensity"],
                           process["AntiSolventDensity"], process["PowerRatio"]])
    if shape <= 0 or np.any(properties[:3] <= 0) or not np.isfinite(properties).all():
        raise ValueError("The shape factor and densities must be positive and finite.")

    pm = np.array(parameters, dtype=float, copy=True)
    if pm.shape != (35,):
        raise ValueError("Pass the 35 IMED kinetic parameters as a one-dimensional array.")
    log_indices = [0, 3, 5, 9, 12, 16, 18, 22, 24, 28]
    pm[log_indices] = 10.**pm[log_indices]
    if not np.isfinite(pm).all():
        raise ValueError("The decoded kinetic parameters must be finite.")
    primary, secondary, growth, dissolution, aggregation, breakage = np.split(pm, [5, 12, 18, 24, 28])
    collisions = prepare_collisions(length, volume, aggregation[2])
    breakup_frequency = np.zeros_like(volume)
    daughters = np.zeros_like(volume)
    if breakage[0] > 0.:
        kb, power, slope, transition, center, deviation, law = breakage
        breakup_frequency = kb * volume**power * np.tanh(slope * (length - transition) + 1)
        if law == 1:
            daughters = 2 / volume
        elif law == 2 and deviation > 0.:
            log_density = -0.5 * ((volume - center) / deviation)**2
            daughters = np.exp(log_density - log_density.max())
        else:
            raise ValueError("Use a supported daughter law and a positive width for Gaussian breakage.")
        daughters /= daughters @ volume

    temperature = np.asarray(experiment["TemperatureData"], dtype=float)
    end = float(np.max(temperature[0]))
    if not np.isfinite(end) or end <= 0:
        raise ValueError("The temperature profile must end at a positive time.")
    temperature = profile_data(temperature, end)
    feed = profile_data(experiment.get("FlowRateData", []), end, default=0.)
    carrier = process["SolvMass"] + process["AntiSolvMass"]
    initial_c = (experiment["InitialConcentration"] if "InitialConcentration" in experiment
                 else np.asarray(experiment["ConcentrationData"]).reshape(2, -1)[1, 0])
    if carrier <= 0 or not 0 <= initial_c < 1:
        raise ValueError("Use positive carrier mass and an API mass fraction in [0, 1).")
    flow = interp1d(*feed, assume_sorted=True, fill_value="extrapolate")
    check_liquid_inventory(feed, flow, end, carrier)

    seed = np.zeros(n)
    if process["SeedMass"]:
        seed_data = np.asarray(experiment["InitialCSDsData"], dtype=float)
        frequency = interp1d(*seed_data, assume_sorted=True, fill_value="extrapolate")(edges)
        frequency = np.where(np.isfinite(frequency) & (frequency >= 0), frequency, 0.)
        number_at_edges = np.r_[0., frequency[1:] / edges[1:]**3]
        seed = (number_at_edges[:-1] + number_at_edges[1:]) / 2
        seed_mass = (volume * width) @ seed * properties[0] / 1e12
        if seed_mass <= 0:
            raise ValueError("The seed CSD has no volume on the mesh.")
        seed *= process["SeedMass"] / seed_mass
    initial = np.r_[seed, initial_c / (1 - initial_c) * carrier, carrier,
                    process["SolvMass"] / carrier, 0., 0., 0.]
    relaxation_time = float(settings.get("EquilibriumRelaxationTime", 1.))
    if not np.isfinite(relaxation_time) or relaxation_time <= 0.:
        raise ValueError("EquilibriumRelaxationTime must be positive and finite in seconds.")
    raw_args = (primary, secondary, growth, dissolution, aggregation, breakage,
                length, width, area, volume, *collisions, breakup_frequency, daughters,
                temperature, feed, [solubility[key] for key in "abcdefgh"], properties, relaxation_time)
    args = PBMArguments(*(jnp.asarray(value) for value in raw_args))
    knots = np.unique(np.r_[temperature[0], feed[0]])
    internal_knots = knots[(knots > 0) & (knots < end)]
    return jnp.asarray(initial), args, jnp.linspace(0, end, 100), jnp.asarray(internal_knots)


def simulate_jax(parameters, experiment, process, solubility, settings=None, *,
                 return_history=False, cancelled=None):
    """Return IMED-style count, concentration, final CSD and optional history."""
    settings = {} if settings is None else settings
    if settings.get("Numerics", "conservative") not in ("conservative", "jax"):
        raise ValueError("This JAX solver implements the conservative IMED scheme.")
    device = resolve_device(settings.get("JaxDevice", "auto"))
    with jax.default_device(device):
        initial, args, time, profile_knots = prepare_jax_case(
            parameters, experiment, process, solubility, settings)
        if cancelled is not None and cancelled():
            raise InterruptedError("PBM simulation stopped.")
        result = run_model(
            initial, args, time, rtol=settings.get("RelTol", 1e-7), atol=settings.get("AbsTol", 1e-9),
            max_step=settings.get("MaxStep"), profile_knots=profile_knots,
            max_steps=settings.get("MaxSteps", 100_000), throw=False, cancelled=cancelled,
        )
        if cancelled is not None and result.result == RESULTS.event_occurred:
            raise InterruptedError("PBM simulation stopped.")
        if result.result != RESULTS.successful:
            steps = int(result.stats["num_steps"])
            raise RuntimeError(f"JAX PBM stopped after {steps:,} integration steps. {RESULTS[result.result]}")
        history = np.array(result.ys).T.copy()
    if not np.isfinite(history).all():
        raise RuntimeError("The JAX PBM produced a nonfinite state.")
    n = args.width.size
    # Match IMED's public history. Internally the solver integrates grams.
    history[n] /= history[n + 1]
    ratio, carrier, solvent = history[n:n + 3]
    liquid_volume = carrier * (solvent / process["SolventDensity"]
                               + (1 - solvent) / process["AntiSolventDensity"])
    density = history[:n]
    roundoff = 64 * np.finfo(density.dtype).eps * np.max(density, axis=0)
    density = np.where((density < 0) & (density >= -roundoff), 0., density)
    outputs = (np.vstack((time, np.asarray(args.width) @ density / liquid_volume)),
               np.vstack((time, ratio / (1 + ratio))),
               np.vstack((args.length, density[:, -1])))
    return (*outputs, history) if return_history else outputs
