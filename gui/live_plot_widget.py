"""Original Matplotlib plots rendered off-thread into a cached Qt image."""
from collections import deque
from copy import deepcopy
import logging
import math

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, Qt
from PySide6.QtGui import QImage, QPainter, QColor
from PySide6.QtWidgets import QWidget

from gui.plot_renderer import OriginalPlotRenderer
from utilities.plot_sampling import extrema_indices

logger = logging.getLogger("orchestrator.plots")


class PlotRenderWorker(QObject):
    frame_ready = Signal(int, object)
    render_failed = Signal(int, str)

    def __init__(self, configs):
        super().__init__()
        self.configs = deepcopy(configs)
        self.renderer = None

    @Slot(object)
    def render(self, request):
        from gui.render_lock import RENDER_LOCK
        with RENDER_LOCK:
            self._render(request)

    def _render(self, request):
        try:
            if QThread.currentThread().isInterruptionRequested():
                return
            if self.renderer is None:
                # Figure, artists and Agg canvas are created and owned here.
                self.renderer = OriginalPlotRenderer(device_configs=self.configs)
            canvas = self.renderer
            if canvas.dark_mode != request["dark"]:
                canvas.set_theme(request["dark"])
            dpi = 100 * request["dpr"]
            canvas.fig.set_dpi(dpi)
            canvas.fig.set_size_inches(request["width"] / dpi, request["height"] / dpi, forward=False)
            canvas.fig.suptitle(request["title"], fontsize=11, fontweight="bold",
                                color="#e5e7eb" if request["dark"] else "#000000")
            times, history = request["times"], request["history"]
            for key, line in (*canvas.lines.items(), *canvas.secondary_lines.items()):
                line.set_data(times, history[key])
            for axis in canvas.axes:
                axis.relim()
                axis.autoscale_view()
            for index, axis in canvas.secondary_axes.items():
                axis.relim()
                if any(config.get("type") == "nidaq_valve" and int(config.get("plot_index", 1)) == index
                       for config in self.configs.values()):
                    axis.set_ylim(-0.05, 1.05)
                    axis.set_yticks([0, 1])
                else:
                    axis.autoscale_view()
            canvas.draw()
            width, height = canvas.get_width_height()
            # Deep-copy Agg's buffer before it can be reused on the next draw.
            image = QImage(canvas.buffer_rgba(), width, height, width * 4,
                           QImage.Format.Format_RGBA8888).copy()
            image.setDevicePixelRatio(request["dpr"])
            self.frame_ready.emit(request["generation"], image)
        except Exception as error:
            self.render_failed.emit(request["generation"], f"{type(error).__name__}: {error}")


class LivePlotWidget(QWidget):
    render_requested = Signal(object)

    def __init__(self, parent=None, device_configs=None, **kwargs):
        super().__init__(parent)
        self.device_configs = deepcopy(device_configs or {})
        self.dark_mode = True
        self.title = "Live telemetry"
        self.t_data = deque(maxlen=2000)
        self.history = {}
        for device_id, config in self.device_configs.items():
            fields = (OriginalPlotRenderer.TELEMETRY_FIELDS.get(config["type"], []) +
                      OriginalPlotRenderer.SECONDARY_FIELDS.get(config["type"], []))
            for field, _ in fields:
                self.history[(device_id, field)] = deque(maxlen=2000)
        self._image = QImage()
        self._pending_data = None
        self._pending_samples = deque()
        self._boundary_flags = deque(maxlen=2000)
        self._generation = 0
        self._dirty = True
        self._rendering = False
        self._closed = False
        self._thread = self._worker = None
        self.render_error = ""
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self.flush_pending)
        self.setMinimumHeight(1000)
        self._schedule()

    def _schedule(self):
        if not self._closed and not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def set_theme(self, dark=True):
        self.dark_mode = bool(dark)
        self._invalidate()

    def _invalidate(self):
        self._generation += 1
        self._dirty = True
        self.render_error = ""
        self._schedule()
        self.update()

    @Slot(int, int)
    def set_experiment_title(self, number, total):
        self.title = f"Experiment {number} of {total}"
        self._pending_data = None
        self._pending_samples.clear()
        self._boundary_flags.clear()
        self.t_data.clear()
        for values in self.history.values():
            values.clear()
        self._image = QImage()
        self._invalidate()

    @staticmethod
    def _number(value):
        try:
            result = float(value)
            return result if math.isfinite(result) else math.nan
        except (ValueError, TypeError):
            return math.nan

    @Slot(dict)
    def append_and_update(self, data):
        if self._closed:
            return
        self._pending_data = data
        for sample in data.get("samples", (data,)):
            if len(self._pending_samples) >= 4096:
                samples = list(self._pending_samples)
                indices = extrema_indices(
                    [[self._number(sample.get(device, {}).get(field)) for sample in samples]
                     for device, field in self.history], len(samples), 2048,
                    [i for i, sample in enumerate(samples) if sample.get("profile_boundary")])
                self._pending_samples = deque(samples[i] for i in indices)
            self._pending_samples.append(sample)
        self._schedule()

    @Slot()
    def flush_pending(self):
        if self._closed:
            return
        self._pending_data = None
        while self._pending_samples:
            data = self._pending_samples.popleft()
            timestamp = self._number(data.get("timestamp"))
            if math.isfinite(timestamp):
                capacity = self.t_data.maxlen or 2000
                if capacity >= 64 and len(self.t_data) >= capacity:
                    flags = list(self._boundary_flags)[-len(self.t_data):]
                    indices = extrema_indices(list(self.history.values()), len(self.t_data), int(capacity * .6),
                                              [i for i, flag in enumerate(flags) if flag])
                    times = list(self.t_data)
                    self.t_data.clear()
                    self.t_data.extend(times[i] for i in indices)
                    for values in self.history.values():
                        previous = list(values)
                        values.clear()
                        values.extend(previous[i] for i in indices)
                    self._boundary_flags.clear()
                    self._boundary_flags.extend(flags[i] for i in indices)
                self.t_data.append(timestamp)
                self._boundary_flags.append(bool(data.get("profile_boundary")))
                for (device_id, field), values in self.history.items():
                    values.append(self._number(data.get(device_id, {}).get(field)))
                self._dirty = True
        if self._dirty and not self._rendering:
            self._dispatch_render()

    def _dispatch_render(self):
        if self._thread is None:
            self._thread = QThread(self)
            self._worker = PlotRenderWorker(self.device_configs)
            self._worker.moveToThread(self._thread)
            self.render_requested.connect(self._worker.render, Qt.ConnectionType.QueuedConnection)
            self._worker.frame_ready.connect(self._accept_frame, Qt.ConnectionType.QueuedConnection)
            self._worker.render_failed.connect(self._render_failed, Qt.ConnectionType.QueuedConnection)
            self._thread.finished.connect(self._worker.deleteLater)
            self._thread.start()
        dpr = self.devicePixelRatioF()
        # Bound image allocations while preserving the aspect ratio.
        dpr = min(dpr, 2400 / max(1, self.width()), 3000 / max(1, self.height()))
        request = {
            "generation": self._generation, "dark": self.dark_mode, "title": self.title,
            "width": max(320, round(self.width() * dpr)),
            "height": max(320, round(self.height() * dpr)), "dpr": dpr,
            "times": tuple(self.t_data),
            "history": {key: tuple(values) for key, values in self.history.items()},
        }
        self._dirty = False
        self._rendering = True
        self.render_requested.emit(request)

    @Slot(int, object)
    def _accept_frame(self, generation, image):
        self._rendering = False
        if self._closed:
            return
        if generation == self._generation:
            self._image = image
            self.render_error = ""
            self.update()
        if self._dirty or self._pending_data is not None:
            self._schedule()

    @Slot(int, str)
    def _render_failed(self, generation, error):
        self._rendering = False
        if self._closed:
            return
        if generation == self._generation:
            self.render_error = error
            logger.error("Plot rendering failed: %s", error)
            self.update()
        if self._dirty or self._pending_data is not None:
            self._schedule()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._invalidate()

    def paintEvent(self, event):
        # This is the only GUI paint work: a cached bitmap, never Matplotlib.
        painter = QPainter(self)
        painter.fillRect(event.rect(), QColor("#111827" if self.dark_mode else "#ffffff"))
        if not self._image.isNull():
            painter.drawImage(self.rect(), self._image)
        else:
            painter.setPen(QColor("#e5e7eb" if self.dark_mode else "#000000"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             self.render_error or "Preparing plots…")

    def shutdown_renderer(self, timeout_ms=5000):
        self._closed = True
        self._refresh_timer.stop()
        self._pending_data = None
        self._pending_samples.clear()
        thread = self._thread
        if thread is not None:
            thread.requestInterruption()
            thread.quit()
            if not thread.wait(timeout_ms):
                return False
            self._thread = self._worker = None
            thread.deleteLater()
        return True

    def closeEvent(self, event):
        if self.shutdown_renderer():
            event.accept()
        else:
            event.ignore()


RealTimePlotCanvas = LivePlotWidget
