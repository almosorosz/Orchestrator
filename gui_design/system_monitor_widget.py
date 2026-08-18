import psutil
import socket
import datetime

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,QHBoxLayout,
    QProgressBar, QTextEdit
)
from PySide6.QtCore import Qt
from gui_design.stylish_widgets import MetricCard
# ==========================================================
# Background worker
# ==========================================================

class MonitorWorker(QObject):

    update = Signal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.old_net = psutil.net_io_counters()
        self.count = 0
        self.disks = ""
        self.processes = ""
        

    @Slot()
    def run(self):

        while self.running:

            self.count += 1

            ram = psutil.virtual_memory()
            net = psutil.net_io_counters()

            data = {
                "name": socket.gethostname(),
                "time": datetime.datetime.now().strftime(
                    "%H:%M:%S"
                ),
                
                "cpu": psutil.cpu_percent(),
                "ram": ram.percent,
                "ram_text":
                    f"{ram.used/1e9:.1f}/"
                    f"{ram.total/1e9:.1f} GB",
                "down":
                    (net.bytes_recv -
                     self.old_net.bytes_recv)
                    /1e6,
                "up":
                    (net.bytes_sent -
                     self.old_net.bytes_sent)
                    /1e6,
                "disk": self.disks,
                "proc": self.processes,
                "temp": "N/A"
            }

            self.old_net = net
            if self.count % 15 == 0:
                self.get_disks()

            if self.count % 5 == 0:
                self.get_processes()

            self.update.emit(data)
            QThread.sleep(2)

    def get_disks(self):

        out = ""

        for d in psutil.disk_partitions():

            try:
                u = psutil.disk_usage(d.mountpoint)
                out += (f"{d.device}: " f"{u.percent}% used\n")

            except:
                
                pass

        self.disks = out

    def get_processes(self):

        p = []

        for x in psutil.process_iter(["name","memory_info"]):
            try:
                p.append((x.info["memory_info"].rss, x.info["name"]))
                
            except:
                pass

        p.sort(reverse=True)
        self.processes = "\n".join(f"{n}: {m/1e6:.0f} MB" for m,n in p[:8])

    def stop(self):
        self.running = False

# ==========================================================
# Dashboard widget
# ==========================================================

class SystemMonitorWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.cpu_card = MetricCard("CPU", "--%", "🖥")
        self.ram_card = MetricCard("RAM", "--%", "💾")
        self.net_card = MetricCard("NET", "--", "🌐")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(self.cpu_card)
        layout.addWidget(self.ram_card)
        layout.addWidget(self.net_card)

        self.thread = QThread(self)
        self.worker = MonitorWorker()

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.update.connect(self.refresh)

        self.thread.start()

    def set_theme(self,if_dark=True):
        self.cpu_card.set_theme(if_dark)
        self.ram_card.set_theme(if_dark)
        self.net_card.set_theme(if_dark)


    @Slot(dict)
    def refresh(self, d):

        cpu = d["cpu"]
        ram = d["ram"]
        net = (f"↓ {d['down']:.2f} MB/s  "f"↑ {d['up']:.2f} MB/s")
        
        self.cpu_card.update_value(f"{cpu:.2f}%", "Normal" if cpu < 80 else "High", "#22c55e" if cpu < 80 else "#facc15")
        self.ram_card.update_value(f"{ram:.2f}%","Normal" if ram < 80 else "High","#22c55e" if ram < 80 else "#facc15")
        self.net_card.update_value(net,"Active" if d["down"] > 0 or d["up"] > 0 else "Idle","#22c55e" if d["down"] > 0 or d["up"] > 0 else "#64748b")

    def closeEvent(self, e):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        e.accept()


