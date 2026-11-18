import numpy as np
from collections import deque
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class OriginalPlotRenderer(FigureCanvas):

    TELEMETRY_FIELDS = {
        "masterflex": [("speed_rpm", "RPM")],
        "jkem": [("speed_rpm", "RPM")],
        "ismatec": [("flow", "Flow")],
        "huber": [("setpoint", "Setpoint"),("temp_internal", "Jacket"),("temp_external", "Process")],
        "mettler": [("weight_g", "Mass")],
        "nidaq_valve": [("measured_voltage", "Voltage")],
        "opcua": [ ("value", "Value")]
    }

    SECONDARY_FIELDS = {
        "jkem": [
            ("torque_ozin", "Torque")
        ],
        "nidaq_valve": [
            ("target_state", "Command")
        ]
    }

    def __init__(self, parent=None, device_configs=None, width=12, height=12, dpi=100):

        self.device_configs = device_configs or {}
        self.dark_mode = True
        self.fig = Figure(
            figsize=(width, height),
            dpi=dpi,
            tight_layout=False
        )
        # Keep the original explicit margins regardless of global rcParams
        # changed by static DoE/analysis figures elsewhere in the application.
        self.fig.set_layout_engine("none")
        
        self.fig.subplots_adjust(
            left=0.07,
            right=0.96,
            bottom=0.055,
            top=0.94,
            wspace=0.35,
            hspace=0.42
        )

        self.axes = [
            self.fig.add_subplot(421),
            self.fig.add_subplot(422),
            self.fig.add_subplot(423),
            self.fig.add_subplot(424),
            self.fig.add_subplot(425),
            self.fig.add_subplot(426),
            self.fig.add_subplot(427),
            self.fig.add_subplot(428)
        ]

        super().__init__(self.fig)

        self.fig.patch.set_facecolor("#111827")

        self.plot_counter = 0
        self.reset_canvas()

    def set_theme(self, dark=True):
    
        self.dark_mode = bool(dark)
    
        if self.dark_mode:
            bg = "#111827"
            fg = "#e5e7eb"
            border = "#475569"
            grid = "#64748b"
        else:
            bg = "#ffffff"
            fg = "#000000"
            border = "#334155"
            grid = "#94a3b8"
    
        self.fig.patch.set_facecolor(bg)
    
        for ax in self.fig.axes:
    
            ax.set_facecolor(bg)
    
            ax.tick_params(
                axis="both",
                colors=fg,
                labelsize=8
            )
    
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
    
            for spine in ax.spines.values():
                spine.set_color(border)
    
            ax.grid(
                True,
                color=grid,
                alpha=0.3,
                linestyle="--"
            )
    
            legend = ax.get_legend()
    
            if legend is not None:
                legend.get_frame().set_facecolor(bg)
                legend.get_frame().set_edgecolor(border)
    
                for label in legend.get_texts():
                    label.set_color(fg)
    
        if self.fig._suptitle is not None:
            self.fig._suptitle.set_color(fg)
    
        self.draw_idle()
        
    def reset_canvas(self):
    
        # Remove twinx axes created during the previous reset
        for ax in list(self.fig.axes):
            if ax not in self.axes:
                ax.remove()
    
        self.t_data = deque(maxlen=2000)
        self.history = {}
        self.lines = {}
        self.secondary_axes = {}
        self.secondary_lines = {}

        for i, ax in enumerate(self.axes, start=1):

            ax.cla()
            ax.set_facecolor("#111827")
            ax.grid(True, alpha=0.25)

            ax.tick_params(
                axis="both",
                colors="#e5e7eb",
                labelsize=8
            )

            ax.xaxis.label.set_color("#e5e7eb")
            ax.yaxis.label.set_color("#e5e7eb")
            ax.title.set_color("#e5e7eb")

            for spine in ax.spines.values():
                spine.set_color("#475569")

            ax.set_title("")

            ax.set_xlabel(
                "Elapsed Time (s)",
                fontsize=7,
                labelpad=2
            )

        for device_id, config in self.device_configs.items():

            device_type = config.get("type")

            fields = self.TELEMETRY_FIELDS.get(
                device_type,
                []
            )

            if not fields:
                continue

            plot_index = int(
                config.get("plot_index", 1)
            )

            plot_index = min(
                max(plot_index, 1),
                8
            )

            ax = self.axes[
                plot_index - 1
            ]

            device_name = config.get(
                "name",
                device_id
            )

            for field, field_label in fields:

                key = (
                    device_id,
                    field
                )

                self.history[key] = deque(maxlen=2000)

                if len(fields) == 1:
                    label = device_name
                else:
                    label = (
                        f"{device_name} - "
                        f"{field_label}"
                    )

                line, = ax.plot(
                    [],
                    [],
                    label=label
                )

                self.lines[key] = line

        for device_id, config in self.device_configs.items():

            device_type = config.get("type")
        
            fields = self.SECONDARY_FIELDS.get(
                device_type,
                []
            )
        
            if not fields:
                continue
        
            plot_index = int(
                config.get("plot_index", 1)
            )
        
            plot_index = min(
                max(plot_index, 1),
                8
            )
        
            ax = self.axes[
                plot_index - 1
            ]
        
            if plot_index not in self.secondary_axes:
        
                ax2 = ax.twinx()
        
                ax2.tick_params(
                    axis="y",
                    colors="#e5e7eb",
                    labelsize=8
                )
        
                ax2.yaxis.label.set_color(
                    "#e5e7eb"
                )
        
                ax2.spines["right"].set_color(
                    "#475569"
                )
        
                self.secondary_axes[
                    plot_index
                ] = ax2
        
            else:
        
                ax2 = self.secondary_axes[
                    plot_index
                ]
        
            device_name = config.get(
                "name",
                device_id
            )
        
            for field, field_label in fields:
        
                key = (
                    device_id,
                    field
                )
        
                self.history[key] = deque(maxlen=2000)
        
                line, = ax2.plot(
                    [],
                    [],
                    "-.",
                    label=(
                        f"{device_name} - "
                        f"{field_label}"
                    )
                )
        
                self.secondary_lines[
                    key
                ] = line

        for plot_index, ax2 in self.secondary_axes.items():
        
            secondary_types = {
                config.get("type")
                for config in self.device_configs.values()
                if int(config.get("plot_index", 1)) == plot_index
            }
        
            if "jkem" in secondary_types:
                ax2.set_ylabel(
                    "Torque (oz-in)",
                    fontsize=8
                )
        
            elif "nidaq_valve" in secondary_types:
            
                ax2.set_ylabel(
                    "Valve command",
                    fontsize=8
                )
            
                ax2.set_ylim(
                    -0.05,
                    1.05
                )
            
                ax2.set_yticks(
                    [0, 1]
                )

        for i, ax in enumerate(self.axes, start=1):
        
            handles, labels = ax.get_legend_handles_labels()
        
            ax2 = self.secondary_axes.get(i)
        
            if ax2 is not None:
        
                handles2, labels2 = (
                    ax2.get_legend_handles_labels()
                )
        
                handles += handles2
                labels += labels2
        
            if handles:
        
                legend = ax.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.01),
                    ncol=min(len(handles), 3),
                    fontsize=6,
                    frameon=True,
                    borderpad=0.25,
                    labelspacing=0.25,
                    handlelength=1.5
                )
        
                for text in legend.get_texts():
                    text.set_color("#e5e7eb")
        
                legend.get_frame().set_facecolor(
                    "#111827"
                )
        
                legend.get_frame().set_edgecolor(
                    "#475569"
                )

        self.set_theme(self.dark_mode)

    def _safe_float(self, value):

        if value is None:
            return np.nan

        try:
            value = float(value)

            return (
                value
                if np.isfinite(value)
                else np.nan
            )

        except (ValueError, TypeError):
            return np.nan

    def _render_sample(self, data: dict):

        t = self._safe_float(
            data.get("timestamp")
        )

        if np.isnan(t):
            return

        self.t_data.append(t)

        for device_id, config in self.device_configs.items():

            device_type = config.get("type")

            fields = self.TELEMETRY_FIELDS.get(
                device_type,
                []
            )

            status = data.get(
                device_id,
                {}
            )

            for field, _ in fields:

                key = (
                    device_id,
                    field
                )

                if key not in self.history:
                    continue

                value = self._safe_float(
                    status.get(field)
                )

                self.history[key].append(
                    value
                )

                self.lines[key].set_data(
                    self.t_data,
                    self.history[key]
                )

        for device_id, config in self.device_configs.items():

            device_type = config.get("type")
        
            fields = self.SECONDARY_FIELDS.get(
                device_type,
                []
            )
        
            status = data.get(
                device_id,
                {}
            )
        
            for field, _ in fields:
        
                key = (
                    device_id,
                    field
                )
        
                if key not in self.history:
                    continue
        
                value = self._safe_float(
                    status.get(field)
                )
        
                self.history[key].append(
                    value
                )
        
                self.secondary_lines[key].set_data(
                    self.t_data,
                    self.history[key]
                )
        
        for ax in self.axes:
        
            ax.relim()
            ax.autoscale_view()
        
        for plot_index, ax2 in self.secondary_axes.items():
        
            secondary_types = {
                config.get("type")
                for config in self.device_configs.values()
                if int(config.get("plot_index", 1)) == plot_index
            }
        
            ax2.relim()
        
            if "nidaq_valve" in secondary_types:
                ax2.set_ylim(-0.05, 1.05)
                ax2.set_yticks([0, 1])
            else:
                ax2.autoscale_view()
        
        self.draw_idle()





    def draw_idle(self):
        pass  # Agg rendering is explicitly requested by the render worker.
