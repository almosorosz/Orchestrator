"""Present stored kernel codes as fixed categorical choices."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox

from .parameter_labels import KERNEL_OPTIONS


def add_kernel_choice(table, row, column, parameter, nominal):
    choice = QComboBox(table)
    for code, label in KERNEL_OPTIONS[parameter].items():
        choice.addItem(label, code)
    selected = choice.findData(nominal)
    if selected < 0:
        choice.addItem(f"Stored value ({nominal:g})", nominal)
        selected = choice.count() - 1
    choice.setCurrentIndex(selected)
    choice.initial_choice = selected
    choice.setToolTip("This choice is fixed during parameter estimation. Lᵢ and Lⱼ are particle sizes; V is fragment volume.")
    table.setCellWidget(row, column, choice)
    for offset in (1, 2, 3):
        item = table.item(row, column + offset)
        item.setText("—")
        item.setFlags(item.flags() & ~(Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsUserCheckable))


def kernel_values(choice):
    code = choice.currentData()
    return (0, code, code, code)
