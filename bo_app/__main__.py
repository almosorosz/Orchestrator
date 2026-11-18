"""Standalone PySide6 Bayesian optimization for CrystalLoop recipes."""
import argparse
import sys

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .optimizer import load_campaign, rank_candidates


class RankJob(QThread):
    ranked = Signal(object, object)
    failed = Signal(str)

    def __init__(self, paths, options, parent):
        super().__init__(parent)
        self.paths = paths
        self.options = options

    def run(self):
        try:
            campaign = load_campaign(*self.paths)
            ranking = rank_candidates(campaign, *self.options)
        except Exception as error:
            self.failed.emit(str(error))
        else:
            self.ranked.emit(campaign, ranking)


class Window(QMainWindow):
    def __init__(self, start_experiment=None):
        super().__init__()
        self.setWindowTitle("CrystalLoop | Bayesian experiment selection")
        self.resize(870, 660)
        self.start_experiment = start_experiment
        self.job = None
        self.campaign = None
        self.ranking = None
        self.profiles = QLineEdit()
        self.outcomes = QLineEdit()
        self.size_mode = QComboBox()
        self.size_mode.addItems(["Minimize", "Maximize", "Target"])
        self.target = self.number(0, 1e6)
        self.target.setEnabled(False)
        self.size_mode.currentTextChanged.connect(
            lambda mode: self.target.setEnabled(mode == "Target"))
        self.weights = [self.number(1, 100) for _ in range(3)]
        self.exploration = self.number(1, 10)
        self.run_button = QPushButton("Rank experiments")
        self.run_button.clicked.connect(self.run_ranking)
        self.export_button = QPushButton("Export next recipe")
        self.export_button.clicked.connect(self.export_recipe)
        self.export_button.setEnabled(False)
        self.start_button = QPushButton("Start next experiment")
        self.start_button.clicked.connect(self.start_recipe)
        self.start_button.setEnabled(False)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Experiment", "Predicted loss", "Uncertainty", "Score"])
        self.status = QLabel("Open generated profiles and the experiment outcomes CSV.")
        self.status.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Generated profiles", self.file_input(self.profiles))
        form.addRow("Experiment outcomes", self.file_input(self.outcomes))
        form.addRow("Crystal size goal", self.size_mode)
        form.addRow("Target size", self.target)
        for label, weight in zip(("Size weight", "Yield weight", "Batch time weight"), self.weights):
            form.addRow(label, weight)
        form.addRow("Exploration weight", self.exploration)
        actions = QHBoxLayout()
        for button in (self.run_button, self.export_button, self.start_button):
            actions.addWidget(button)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.addLayout(form)
        layout.addLayout(actions)
        layout.addWidget(self.table)
        layout.addWidget(self.status)
        self.setCentralWidget(body)

    @staticmethod
    def number(value, maximum):
        widget = QDoubleSpinBox()
        widget.setRange(0, maximum)
        widget.setValue(value)
        return widget

    def file_input(self, field):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("Browse…")

        def choose():
            path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV (*.csv)")
            if path:
                field.setText(path)

        button.clicked.connect(choose)
        layout.addWidget(field)
        layout.addWidget(button)
        return widget

    def run_ranking(self):
        self.run_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.table.setRowCount(0)
        self.status.setText("Training trajectory encoders and ranking candidates…")
        options = (self.size_mode.currentText(), self.target.value(),
                   [w.value() for w in self.weights], self.exploration.value())
        self.job = RankJob((self.profiles.text(), self.outcomes.text()), options, self)
        self.job.ranked.connect(self.show_ranking)
        self.job.failed.connect(self.status.setText)
        self.job.finished.connect(self.finish_ranking)
        self.job.start()

    def finish_ranking(self):
        self.job.deleteLater()
        self.job = None
        self.run_button.setEnabled(True)

    def closeEvent(self, event):
        if self.job is not None:
            event.ignore()
        else:
            super().closeEvent(event)

    def show_ranking(self, campaign, ranking):
        self.campaign, self.ranking = campaign, ranking
        self.table.setRowCount(len(ranking))
        for row, (name, *scores) in enumerate(ranking.itertuples(index=False, name=None)):
            for column, value in enumerate([str(name), *(f"{v:.4g}" for v in scores)]):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        self.export_button.setEnabled(True)
        self.start_button.setEnabled(self.start_experiment is not None)
        name, _, amount = self.next_recipe()
        handoff = "" if self.start_experiment else " Export the recipe to hand it off to CrystalLoop."
        self.status.setText(f"Next experiment: {name}. Seed amount: {amount:g}.{handoff}")

    def next_recipe(self):
        name = self.ranking.iloc[0]["experiment"]
        profiles = self.campaign.profiles
        recipe = profiles.loc[profiles["experiment"] == name].drop(columns="experiment").copy()
        return name, recipe, float(self.campaign.outcomes.loc[name, "seed_amount"])

    def export_recipe(self):
        name, recipe, amount = self.next_recipe()
        path, _ = QFileDialog.getSaveFileName(self, "Export next recipe", "bo_next.csv", "CSV (*.csv)")
        if path:
            try:
                recipe.assign(experiment=name, seed_amount=amount).to_csv(path, index=False)
            except OSError as error:
                self.status.setText(f"Unable to export recipe. {error}")
            else:
                self.status.setText(f"Exported {name} to {path}")

    def start_recipe(self):
        name, recipe, amount = self.next_recipe()
        try:
            self.start_experiment(name, recipe, amount)
        except Exception as error:
            self.status.setText(f"Start failed. {error}")
        else:
            self.start_button.setEnabled(False)
            self.run_button.setEnabled(False)
            self.status.setText(f"Started {name}. Reopen the campaign with updated experiment statuses.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="?")
    parser.add_argument("outcomes", nargs="?")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = Window()
    if args.profiles:
        window.profiles.setText(args.profiles)
    if args.outcomes:
        window.outcomes.setText(args.outcomes)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
