from PySide6.QtWidgets  import (QDialog, QWidget, QFormLayout, 
                                QGroupBox,QVBoxLayout,QCheckBox,QSpinBox,
                                QDoubleSpinBox,QLineEdit,QDialogButtonBox,
                                QVBoxLayout,QScrollArea)


from dataclasses import fields, is_dataclass

class DataclassEditor(QWidget):

    def __init__(self, obj):
        super().__init__()

        self.obj = obj
        self.widgets = {}

        layout = QFormLayout(self)

        for f in fields(obj):

            value = getattr(obj, f.name)

            # nested dataclass
            if is_dataclass(value):

                editor = DataclassEditor(value)

                box = QGroupBox(f.name)
                v = QVBoxLayout(box)
                v.addWidget(editor)

                layout.addRow(box)

                self.widgets[f.name] = editor
                continue

            # bool
            if isinstance(value, bool):
                w = QCheckBox()
                w.setChecked(value)

            # int
            elif isinstance(value, int):
                w = QSpinBox()
                w.setRange(-999999,999999)
                w.setValue(value)

            # float
            elif isinstance(value, float):
                w = QDoubleSpinBox()
                w.setDecimals(4)
                w.setRange(-1e6,1e6)
                w.setValue(value)

            # string
            elif isinstance(value, str):
                w = QLineEdit(value)

            else:
                continue

            self.widgets[f.name] = w
            layout.addRow(f.name, w)
            
    def save(self):

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