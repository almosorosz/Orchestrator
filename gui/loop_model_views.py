"""Present the automatic worker's inputs, progress, and results in the modeling UI."""
from imed_python.design_dialog import DesignDialog
from imed_python.estimation_dialog import ParameterEstimation


class LoopModelViews:
    def __init__(self, window):
        self.window = window
        self.input_dialog = None
        self.readout = None

    def close_dialogs(self, *, keep_readout=False):
        for dialog in (self.input_dialog, None if keep_readout else self.readout):
            if dialog is not None:
                dialog.close()
                dialog.deleteLater()
        self.input_dialog = None
        if not keep_readout:
            self.readout = None

    def close(self, *, keep_readout=False):
        self.close_dialogs(keep_readout=keep_readout)
        self.window.bo_page.loop_busy = False
        self.window.bo_page.update_buttons()

    def _open_page(self, page):
        self.window.switch_page(self.window.stack.indexOf(page))

    def progress(self, message, mode):
        if mode == "bo":
            self.window.bo_page.status.setText(message)
        elif isinstance(self.readout, ParameterEstimation):
            self.readout.log_progress(message)
        elif self.readout is not None:
            self.readout.status.setText(message)

    def show_stage(self, stage, value, job):
        if stage == "bo_inputs":
            view = self.window.bo_page
            last_proposal = view.proposal if job.collect_only else None
            view.loop_busy = True
            view.restore(value)
            if last_proposal is not None:
                view.set_proposal(last_proposal)
            view.tabs.setCurrentIndex(0)
            self._open_page(view)
            self.window.log_message(f"Opened BO with {len(value['runs'])} experiment records.")
            return
        if stage == "bo_result":
            view = self.window.bo_page
            view.set_proposal(value)
            view.tabs.setCurrentIndex(view.tabs.count() - 1)
            self.window.log_message("BO predictions and proposed recipe are displayed.")
            return
        if stage == "bo_stage":
            view = self.window.bo_page
            view.state["loop_stages"].append(value)
            view.update_buttons()
            return

        page = self.window.imed_page
        panel = page.panel
        if stage == "imed_inputs":
            self.close_dialogs()
            panel.display_loop_case(value["case"], value["folder"])
            panel.show_message(f"Automatic LOOP imported {len(value['case']['ExpData'])} experiments into IMED.")
            self._open_page(page)
            self.window.log_message(f"Updated IMED Experimental Input with {len(value['case']['ExpData'])} runs.")
            if not job.collect_only:
                self.input_dialog = panel.show_inputs(modeless=True)
        elif stage == "imed_pe":
            dialog = ParameterEstimation(value, self.window, workers=job.session.workers)
            for widget in (dialog.run_button, dialog.advanced_button, dialog.load_settings_button,
                           dialog.save_case_button, dialog.save_settings_button, dialog.uncertainty_button,
                           dialog.model, dialog.terms, dialog.size_samples, *dialog.weights):
                widget.setEnabled(False)
            dialog.stop_button.clicked.disconnect(dialog.stop_fit)
            self._show_readout(dialog, f"Automatic parameter estimation is fitting {len(value['ExpData'])} experiments.")
            self.window.log_message("Opened IMED Parameter Estimation with the saved fit settings.")
        elif stage == "imed_generation" and isinstance(self.readout, ParameterEstimation):
            self.readout.optimization_progress.append(*value)
        elif stage == "imed_fits" and isinstance(self.readout, ParameterEstimation):
            panel.display_loop_case(value["case"], job.manifest["campaign_folder"])
            self.readout.case = value["case"]
            self.readout.show_results(value["fits"], {}, remember=False, screening=value["screening"])
            self.readout.stop_button.setEnabled(False)
            self.readout.details_button.setEnabled(True)
            self.readout.save_case_button.setEnabled(True)
            panel.show_message("Automatic parameter estimation completed. Fitted models saved to the MODEL folder.")
        elif stage == "imed_design":
            self.close_dialogs()
            panel.display_loop_case(value["case"], job.manifest["campaign_folder"])
            optimal = value["method"] == "D-optimal design"
            dialog = DesignDialog(value["case"], optimal, self.window, model_indices=value["survivors"],
                                  workers=job.session.workers)
            dialog.load_request(value["request"])
            dialog.run_button.setEnabled(False)
            dialog.stop_button.clicked.disconnect(dialog.stop_design)
            self._show_readout(dialog, f"Automatic {value['method'].lower()} is running.")
            self.window.log_message(f"Opened the IMED {value['method'].lower()} window.")
        elif stage == "imed_result":
            panel.display_loop_case(value["case"], job.manifest["campaign_folder"])
            if isinstance(self.readout, DesignDialog):
                self.readout.show_recipe(value["result"], remember=False)
                self.readout.export_button.setEnabled(True)
                self.readout.stop_button.setEnabled(False)
                self.readout.execute_button.setVisible(True)
                self.readout.execute_button.setToolTip("The automatic loop will execute this recipe.")
                self.window.log_message("IMED recommended profiles are displayed.")

    def _show_readout(self, dialog, message):
        self.readout = dialog
        dialog.stop_button.clicked.connect(self.window.stop_command)
        dialog.stop_button.setEnabled(True)
        dialog.status.setText(message)
        dialog.show()
        dialog.raise_()
