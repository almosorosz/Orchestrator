"""One shared DoE-style renderer for component, timeline and manual profiles."""
import math

from matplotlib.figure import Figure


COLORS = {"T": "#95979c", "AS": "#ef4444", "seed": "#3b82f6", "solution": "#f59e0b",
          "solvent": "#d946ef", "clean_sol": "#06b6d4", "valve_state": "#eab308",
          "stirring_rate": "#22c55e", "stirring_rate_seed": "#14b8a6", "stirring_rate_sol": "#a78bfa"}
STYLES = {"AS": "--", "solution": "-.", "solvent": "-.", "clean_sol": "-."}


def profile_figure(signals, index=1, name=""):
    """Signals carry their own times (hours), units, labels and semantic roles."""
    figure = Figure(figsize=(max(10, 6 + .6 * len(signals)), 4), dpi=100)
    figure.patch.set_facecolor("none")
    main = figure.add_subplot(111)
    axes, handles = {}, []
    for position, signal in enumerate(signals):
        key, role = signal["key"], signal["role"]
        axis = main if position == 0 else main.twinx()
        if position > 1:
            axis.spines["right"].set_position(("outward", 45 * (position - 1)))
        color = COLORS.get(role, "#94a3b8")
        axis.set_facecolor("none")
        axis.tick_params(axis="both", colors="#94a3b8", labelsize=8, length=4)
        for spine in axis.spines.values():
            spine.set_color("#16263d")
            spine.set_linewidth(.8)
        axis.spines["top"].set_visible(False)
        axis.spines["left" if position == 0 else "right"].set_color(color)
        axis.tick_params(axis="y", colors=color)
        line, = axis.plot(signal["times"], signal["values"], color=color,
                          linewidth=2. if role == "T" else 1.5 if role == "valve_state" else 1.7,
                          linestyle=STYLES.get(role, "-"), label=signal["label"],
                          drawstyle="steps-post" if role == "valve_state" else "default")
        label = signal["label"] + (f" [{signal['unit']}]" if signal.get("unit") else "")
        axis.set_ylabel(label, color=color, fontsize=9 if position == 0 else 8)
        axes[key] = axis
        handles.append(line)
    main.grid(True, which="major", axis="both", color="#24364d", linestyle="--", linewidth=.7, alpha=.7)
    main.set_axisbelow(True)
    main.set_xlabel("Time (h)", color="#95979c", fontsize=9, labelpad=8)
    main.text(1., 1.04, f"EXP {index:02d}" + (f" · {name}" if name else ""), transform=main.transAxes,
              ha="right", va="bottom", color="#94a3b8", fontsize=8, fontweight="bold")
    legend = main.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.25), ncol=4,
                         fontsize=8, frameon=True, fancybox=True, framealpha=.95, borderpad=.7,
                         columnspacing=1.4, handlelength=2.5)
    legend.get_frame().set_facecolor("none")
    legend.get_frame().set_edgecolor("#16263d")
    for text in legend.get_texts():
        text.set_color("#95979c")
    right_margin = (.8 + .625 * max(0, len(signals) - 2)) / figure.get_figwidth()
    figure.subplots_adjust(left=.08, right=max(.25, 1 - right_margin),
                           top=.82 - min(.12, max(0, math.ceil(len(signals) / 4) - 2) * .04), bottom=.15)
    return figure, axes


def manual_profile_figures(experiments, configs):
    """Preview saved Control Panel experiments without connecting any devices."""
    from experiment.models import ExperimentDefinition
    result = []
    for index, experiment in enumerate(experiments, 1):
        definition = (experiment if isinstance(experiment, ExperimentDefinition) else
                      ExperimentDefinition.from_frame(experiment, configs, name=f"Manual experiment {index}"))
        signals = []
        for profile in definition.profiles:
            config = configs[profile.device_id]
            role = {"huber": "T", "jkem": "stirring_rate", "nidaq_valve": "valve_state"}.get(config["type"])
            if role is None:
                role = {"PMP_AS": "AS", "PAS1": "AS", "PMP_SOLU": "solution", "PSOLU1": "solution",
                        "PMP_SOLV": "solvent", "PSOLV1": "solvent", "PMP_CLEN": "clean_sol", "PCLEN1": "clean_sol",
                        "PMP_SEED": "seed", "PSEED1": "seed"}.get(profile.device_id, "pump")
            signals.append({"key": profile.device_id, "role": role,
                            "label": f"{profile.device_id} · {config.get('name', profile.device_id)}",
                            "unit": profile.unit, "times": [p.time_s / 3600 for p in profile.points],
                            "values": [p.target for p in profile.points]})
        signals.sort(key=lambda signal: signal["role"] != "T")
        result.append(profile_figure(signals, index, definition.name)[0])
    return result
