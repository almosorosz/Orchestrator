
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets  import (QDialog, QWidget, QFormLayout, QTableWidget,
                                QGroupBox,QVBoxLayout,QCheckBox,QSpinBox,
                                QDoubleSpinBox,QLineEdit,QDialogButtonBox,
                                QVBoxLayout,QScrollArea, QSizePolicy)


from dataclasses import fields, is_dataclass

from dataclasses import fields, is_dataclass, dataclass, field
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QDialog, QWidget, QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QVBoxLayout, QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QDialogButtonBox, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt

class DataclassEditor(QWidget):
    def __init__(self, obj):
        super().__init__()
        self.obj = obj
        self.widgets = {}
        
        # Primary container layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Form layout for primitive standard variables (floats, ints, bools, etc.)
        self.form_widget = QWidget()
        self.form_layout = QFormLayout(self.form_widget)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        
        # Separate list to collect nested DefineProcessParameter objects for the matrix view
        self.process_params_fields = []
        
        # Categorize fields
        for f in fields(obj):
            value = getattr(obj, f.name)
            
            if is_dataclass(value):
                if type(value).__name__ == "DefineProcessParameter":
                    # Collect parameters for the dense matrix layout
                    self.process_params_fields.append((f.name, value))
                else:
                    # Handle other nested dataclasses normally
                    editor = DataclassEditor(value)
                    box = QGroupBox(f.name)
                    v = QVBoxLayout(box)
                    v.addWidget(editor)
                    self.main_layout.addWidget(box)
                    self.widgets[f.name] = editor
                continue

            # Handle standard fields
            if isinstance(value, bool):
                w = QCheckBox()
                w.setChecked(value)
            elif isinstance(value, int):
                w = QSpinBox()
                w.setRange(-999999, 999999)
                w.setValue(value)
            elif isinstance(value, float):
                w = QDoubleSpinBox()
                w.setDecimals(4)
                w.setRange(-1e6, 1e6)
                w.setValue(value)
            elif isinstance(value, str):
                w = QLineEdit(value)
            else:
                continue
                
            self.widgets[f.name] = w
            self.form_layout.addRow(f.name, w)

        # Append standard form fields to main layout if they exist
        if self.form_layout.rowCount() > 0:
            self.main_layout.addWidget(self.form_widget)

        # Build the matrix grid if DefineProcessParameter items were captured
        if self.process_params_fields:
            self.setup_matrix_view()

    def setup_matrix_view(self):
        """Creates a compact table matrix for DefineProcessParameter dataclasses."""
        group_box = QGroupBox("Experimental Process Parameters Matrix")
        group_layout = QVBoxLayout(group_box)
        
        # Setup table configuration
        self.table = QTableWidget()
        self.table.setRowCount(len(self.process_params_fields))
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Value", "Min Value", "Max Value", "In DoE?", "Only Min/Max?"
        ])
        
        # Structure formatting settings
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        
        row_labels = []
        self.matrix_widgets = {}  # Keep tracking pointers for saving state
        
        for row_idx, (field_name, param_obj) in enumerate(self.process_params_fields):
            row_labels.append(field_name)
            self.matrix_widgets[field_name] = {"obj": param_obj}
            
            # 1. Value (DoubleSpinBox)
            sb_val = QDoubleSpinBox()
            sb_val.setDecimals(4)
            sb_val.setRange(-1e6, 1e6)
            sb_val.setValue(param_obj.value)
            self.table.setCellWidget(row_idx, 0, sb_val)
            self.matrix_widgets[field_name]["value"] = sb_val
            
            # 2. Min Value (DoubleSpinBox)
            sb_min = QDoubleSpinBox()
            sb_min.setDecimals(4)
            sb_min.setRange(-1e6, 1e6)
            sb_min.setValue(param_obj.min_val)
            self.table.setCellWidget(row_idx, 1, sb_min)
            self.matrix_widgets[field_name]["min_val"] = sb_min
            
            # 3. Max Value (DoubleSpinBox)
            sb_max = QDoubleSpinBox()
            sb_max.setDecimals(4)
            sb_max.setRange(-1e6, 1e6)
            sb_max.setValue(param_obj.max_val)
            self.table.setCellWidget(row_idx, 2, sb_max)
            self.matrix_widgets[field_name]["max_val"] = sb_max
            
            # 4. In DoE (Centered CheckBox)
            cb_doe = QCheckBox()
            cb_doe.setChecked(param_obj.if_in_doe)
            container_doe = QWidget()
            layout_doe = QVBoxLayout(container_doe)
            layout_doe.addWidget(cb_doe, 0, Qt.AlignmentFlag.AlignCenter)
            layout_doe.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(row_idx, 3, container_doe)
            self.matrix_widgets[field_name]["if_in_doe"] = cb_doe
            
            # 5. Only Min/Max (Centered CheckBox)
            cb_mm = QCheckBox()
            cb_mm.setChecked(param_obj.if_only_minmax)
            container_mm = QWidget()
            layout_mm = QVBoxLayout(container_mm)
            layout_mm.addWidget(cb_mm, 0, Qt.AlignmentFlag.AlignCenter)
            layout_mm.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(row_idx, 4, container_mm)
            self.matrix_widgets[field_name]["if_only_minmax"] = cb_mm

        self.table.setVerticalHeaderLabels(row_labels)
        
        # Dynamically set fixed height to eliminate matrix scrollbars inside the view
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table_height = (self.table.horizontalHeader().height() + 
                        sum(self.table.rowHeight(i) for i in range(self.table.rowCount())) + 15)
        self.table.setMinimumHeight(min(table_height, 500))
        
        group_layout.addWidget(self.table)
        self.main_layout.addWidget(group_box)

    def save(self):
        # Save structural changes for primitive variables
        for name, widget in self.widgets.items():
            if isinstance(widget, DataclassEditor):
                widget.save()
            elif isinstance(widget, QCheckBox):
                setattr(self.obj, name, widget.isChecked())
            elif isinstance(widget, QSpinBox):
                setattr(self.obj, name, widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                setattr(self.obj, name, widget.value())
            elif isinstance(widget, QLineEdit):
                setattr(self.obj, name, widget.text())
                
        # Save structured changes from the Dense Table Matrix
        if hasattr(self, 'matrix_widgets'):
            for field_name, controls in self.matrix_widgets.items():
                param_obj = controls["obj"]
                param_obj.value = controls["value"].value()
                param_obj.min_val = controls["min_val"].value()
                param_obj.max_val = controls["max_val"].value()
                param_obj.if_in_doe = controls["if_in_doe"].isChecked()
                param_obj.if_only_minmax = controls["if_only_minmax"].isChecked()
                
                # Update back onto parent reference
                setattr(self.obj, field_name, param_obj)


class ExperimentProfileDialog(QDialog):

    def __init__(self, profile, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Experiment Profile")
        self.resize(900,700)

        self.editor = DataclassEditor(profile)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok |
            QDialogButtonBox.Cancel
        )

        buttons.accepted.connect(self.accept_changes)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.editor)

        layout.addWidget(scroll)
        layout.addWidget(buttons)

    def accept_changes(self):
        self.editor.save()
        self.accept()


class DOEPlotsDialog(QDialog):

    def __init__(self, figures, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("DoE Experiment Profiles")
        self.resize(1000, 800)

        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        container_layout = QVBoxLayout(container)

        scroll.setWidget(container)

        layout.addWidget(scroll)

        self.canvases = []

        for i, fig in enumerate(figures):

            canvas = FigureCanvas(fig)

            canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed
            )

            canvas.setMinimumHeight(350)

            container_layout.addWidget(canvas)

            self.canvases.append(canvas)

        container_layout.addStretch()