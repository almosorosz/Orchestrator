"""Display the exact standalone notebook renderer in an embedded web view."""
from pathlib import Path
from PySide6.QtCore import Qt, QUrl, QTemporaryDir
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
from app.paths import DATA_DIR
from notebooks.MSI_graph_compatible import MSI_encoder_w_modules


class SystemGraphView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.web = QWebEngineView(self)
        self.web.setStyleSheet('background: transparent;')
        self.web.page().setBackgroundColor(Qt.GlobalColor.transparent)
        self.web.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
        layout.addWidget(self.web)
        self.setMinimumHeight(320)
        self.graph_data = None
        self.module_colors = {}
        self.network = None
        self.labels_visible = True
        self._ready = False
        self.web.loadFinished.connect(self._loaded)
        directory = DATA_DIR / 'runtime'
        directory.mkdir(parents=True, exist_ok=True)
        self._temporary = QTemporaryDir(str(directory / 'msi_graph_XXXXXX'))
        self.html_file = Path(self._temporary.path()) / 'graph.html'

    def draw_graph(self, graph, code):
        self.graph_data = graph
        encoder = MSI_encoder_w_modules(code)
        self.network = encoder.build_network(font_color=self.palette().windowText().color().name())
        self.module_colors = dict(encoder._module_colors)
        document, _ = encoder._document(self.network)
        self.html_file.write_text(document, encoding='utf-8')
        self._ready = False
        self.web.load(QUrl.fromLocalFile(str(self.html_file)))

    def _loaded(self, okay):
        self._ready = okay
        if okay:
            self.show_labels(self.labels_visible)
            self.fit_graph()

    def clear(self):
        self.graph_data = self.network = None
        self.module_colors = {}
        self._ready = False
        self.web.setHtml('<html><body></body></html>')

    def auto_layout(self):
        if self._ready:
            self.web.page().runJavaScript('if(window.msiAutoArrange){window.msiAutoArrange();}')

    def show_labels(self, visible):
        self.labels_visible = visible
        if self._ready:
            self.web.page().runJavaScript('if(window.network){network.setOptions({nodes:{font:{size:' +
                                         ('14' if visible else '0') + '}}});}')

    def fit_graph(self):
        if self._ready:
            self.web.page().runJavaScript('if(window.network){network.fit({animation:false});}')

    def save_image(self, filename):
        return self.web.grab().save(str(filename), 'PNG')
