import psutil
import socket
import datetime

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,QHBoxLayout,
    QProgressBar, QTextEdit
)
from PySide6.QtCore import Qt
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

    def __init__(self,parent=None):

        super().__init__(parent)

        self.cpu = QProgressBar()
        self.ram = QProgressBar()
        self.info = QLabel()
        self.net = QLabel()
        self.disk = QTextEdit()
        self.disk.setMaximumHeight(60)
        self.proc = QTextEdit()
        self.proc.setMaximumHeight(80)

        layout = QHBoxLayout(self)


        self.cpu_label = QLabel("CPU\n--%")
        self.ram_label = QLabel("RAM\n--%")
        self.net_label = QLabel("NET\n--")
        self.disk_label = QLabel("DISK\n--")
        
        
        for w in [
            self.cpu_label,
            self.ram_label,
            self.net_label,
            self.disk_label
        ]:
            w.setAlignment(
                Qt.AlignCenter
            )
            layout.addWidget(w)

        self.thread = QThread(self)
        self.worker = MonitorWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.update.connect(self.refresh)
        self.thread.start()

    @Slot(dict)
    def refresh(self,d):
    
        self.cpu_label.setText(
            f"CPU\n{d['cpu']}%"
        )
    
        self.ram_label.setText(
            f"RAM\n{d['ram']}%"
        )
    
        self.net_label.setText(
            f"NET\n↓{d['down']:.1f}\n↑{d['up']:.1f}"
        )
    
        self.disk_label.setText(
            d["disk"].split("\n")[0]
        )
    

    def closeEvent(self,e):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        e.accept()