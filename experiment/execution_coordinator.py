from PySide6.QtCore import QObject, QTimer, Signal

from app.controller import Stage
from pathlib import Path
import json
import logging

logger = logging.getLogger("orchestrator.execution")

class ExperimentCoordinator(QObject):

    omni_completed = Signal(dict)
    model_completed = Signal(dict)
    next_experiment_required = Signal(dict)
    stage_changed = Signal(object)
    experiment_progress = Signal(dict)
    error_occurred = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # ------------------------------------------------------
        # EXECUTION OBJECTS
        # ------------------------------------------------------

        self.controller = None
        self.data_processor = None
        self.experiment_creator = None
        # An optimizer adapter must explicitly provide the next experiment.
        self.next_experiment_provider = None
        self.campaign_mode = False
        self.awaiting_campaign = False
        self.campaign_id = None
        self.expected_experiments = set()
        self.completed_experiments = set()
        self._waiting_files = set()

        # ------------------------------------------------------
        # MONITORING STATE
        # ------------------------------------------------------

        self.monitor_mode = None
        self.monitor_path = None
        self.monitor_num = None

        # ------------------------------------------------------
        # COUNTERS
        # ------------------------------------------------------

        self.conducted_experiments_count = 0
        self.previous_exp_count = 0
        self._seen_omni_files = set()

        self.model_run_count = 0
        self.omni_run_count = 0

        # ------------------------------------------------------
        # TIMER
        # ------------------------------------------------------

        self.timer = QTimer(self)

        self.timer.timeout.connect(self.monitoring_tick)

    def _defer_file(self, path, error):
        path = str(path)
        if path not in self._waiting_files:
            logger.warning("Monitoring is waiting for a recording file to become readable; retrying automatically: %s (%s). "
                           "If it is open in another application, close it. Experiments continue.", path, error)
            self._waiting_files.add(path)

    def _read_campaign_json(self, path):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, FileNotFoundError, json.JSONDecodeError) as error:
            # Atomic publication and cloud synchronization can briefly hide,
            # lock or expose an incomplete marker. Retry on the next timer tick.
            self._defer_file(path, error)
            return None
        if str(path) in self._waiting_files:
            self._waiting_files.discard(str(path))
            logger.info("Monitoring file readable again: %s", path)
        return payload


    def set_controller(self, controller):
        self.controller = controller


    def set_data_processor(self, data_processor):
        self.data_processor = data_processor


    def set_experiment_creator(
        self,
        experiment_creator,
    ):
        self.experiment_creator = (
            experiment_creator
        )

    def handle_monitoring_error(self, error):
        """
        Stop execution after an unexpected monitoring failure
        and notify the GUI.
        """
    
        stage = self.monitor_mode
        path = self.monitor_path
    
        error_info = {
            "stage": stage.value if stage is not None else "None",
            "path": str(path) if path is not None else "",
            "error_type": type(error).__name__,
            "message": str(error),
        }
    
        # Stop the timer immediately.
        self.stop_timer()
    
        # Stop the controller, preserving the existing stop API.
        if self.controller is not None:
            try:
                self.controller.stop()
            except Exception as stop_error:
                error_info["message"] += (
                    f"; Controller stop failed: {stop_error}"
                )
    
        # Execution is no longer active.
        self.stage_changed.emit(None)
    
        # Report the original failure after stopping.
        self.error_occurred.emit(error_info)

    def set_monitor(
        self,
        mode,
        path,
        num=None,
    ):
        self.monitor_mode = mode
        self.monitor_path = path
        self.monitor_num = num


    def start_timer(self, interval_ms):
        self.timer.start(interval_ms)


    def stop_timer(self):
        self.timer.stop()


    def reset(self):
        """
        Stop execution and restore coordinator state.
        """
    
        # Stop Controller + timer first.
        self.stop_execution()
        self.next_experiment_provider = None
        self.campaign_mode = False
        self.awaiting_campaign = False
        self.campaign_id = None
        self.expected_experiments.clear()
        self.completed_experiments.clear()
        self._waiting_files.clear()
    
        # ------------------------------------------------------
        # EXECUTION OBJECTS
        # ------------------------------------------------------
    
        self.controller = None
        self.data_processor = None
        self.experiment_creator = None
    
        # ------------------------------------------------------
        # MONITORING STATE
        # ------------------------------------------------------
    
        self.monitor_mode = None
        self.monitor_path = None
        self.monitor_num = None
    
        # ------------------------------------------------------
        # COUNTERS
        # ------------------------------------------------------
    
        self.conducted_experiments_count = 0
        self.previous_exp_count = 0
    
        self.model_run_count = 0
        self.omni_run_count = 0

    def monitoring_tick(self):
        """
        Poll the active execution directory.
    
        Temporary filesystem locks/disappearances wait for the next tick.
        Unexpected failures stop execution and emit error_occurred.
        """
    
        if self.controller is None:
            self.stop_timer()
            return
    
        if self.controller.process_stop:
            self.stop_timer()
            return
    
        try:
    
            # --------------------------------------------------
            # OMNI
            # --------------------------------------------------
    
            if self.monitor_mode == Stage.OMNI:

                if self.campaign_mode:
                    self.poll_campaign()
                    return
    
                # Visible experiment count means detected CSV files, as in
                # the original workflow. Stability gates stage completion only.
                detected = set(self.controller.detected_omni_files(self.monitor_path))
                current_count = len(detected)
                new_experiments = len(detected - self._seen_omni_files)
                self._seen_omni_files.update(detected)
                self.conducted_experiments_count += new_experiments
                self.previous_exp_count = current_count
    
                if new_experiments > 0:
    
                    self.experiment_progress.emit({
                        "new_experiments": new_experiments,
                        "total_experiments": (
                            self.conducted_experiments_count
                        ),
                        "current_batch_count": current_count,
                    })
    
                self.check_omni_transition(
                    self.controller.model_folder
                )
    
            # --------------------------------------------------
            # MODEL
            # --------------------------------------------------
    
            elif self.monitor_mode == Stage.MODEL:
    
                self.check_model_transition(
                    self.controller.omni_folder
                )
    
        except (PermissionError, FileNotFoundError) as error:
            self._defer_file(error.filename or self.monitor_path, error)
        except Exception as error:
    
            self.handle_monitoring_error(error)

    def record_omni_file_count(self, current_count):
        """
        Update the cumulative experiment count from the actual
        number of OMNI result files detected.
        """
    
        # Protect against a folder/count reset.
        if current_count < self.previous_exp_count:
            self.previous_exp_count = current_count
            return 0
    
        new_experiments = (
            current_count
            - self.previous_exp_count
        )
    
        if new_experiments > 0:
            self.conducted_experiments_count += (
                new_experiments
            )
    
        self.previous_exp_count = current_count
    
        return new_experiments

    def begin_campaign(self, campaign):
        if self.controller is None:
            raise RuntimeError("Define experiment and MODEL folders first")
        self.campaign_mode = True
        self.awaiting_campaign = False
        self.campaign_id = campaign.campaign_id
        self.expected_experiments = {e.experiment_id for e in campaign.experiments}
        if not self.expected_experiments or len(self.expected_experiments) != len(campaign.experiments):
            raise ValueError("Campaign experiments must have distinct identities")
        self.completed_experiments = set()
        self._waiting_files.clear()
        self.conducted_experiments_count = 0
        self.controller.exp_num(len(self.expected_experiments))
        self.monitor_path = None
        self.monitor_mode = Stage.OMNI
        self.controller.start()
        self.stage_changed.emit(Stage.OMNI)
        self.start_timer(500)

    def set_campaign_folder(self, folder):
        if not self.campaign_mode or self.controller.process_stop:
            return
        path = Path(folder).resolve()
        if path.parent != Path(self.controller.omni_folder).resolve():
            raise ValueError("Campaign recording folder is outside the selected experiment folder")
        self.monitor_path = str(path)
        self.controller.current_omni_folder = str(path)

    def poll_campaign(self):
        if self.monitor_path is None:
            return
        folder = Path(self.monitor_path)
        if not (folder / "definition.json").exists():
            return  # The recording writer may still be committing startup metadata.
        definition = self._read_campaign_json(folder / "definition.json")
        if definition is None:
            return
        if definition["campaign"]["campaign_id"] != self.campaign_id:
            raise ValueError("Campaign recording identity does not match the active loop")
        completed = set()
        skipped = set()
        for directory in folder.iterdir():
            marker = directory / "run.json"
            if not directory.is_dir() or not marker.exists():
                continue
            run = self._read_campaign_json(marker)
            if run is None:
                continue
            if run.get("run_id") == directory.name and run.get("experiment_id") in self.expected_experiments:
                if run.get("recording_complete"):
                    if run.get("status") == "completed":
                        completed.add(run["experiment_id"])
                    elif run.get("status") == "skipped":
                        skipped.add(run["experiment_id"])
        new = completed - self.completed_experiments
        self.completed_experiments.update(new)
        self.conducted_experiments_count = len(self.completed_experiments)
        if new:
            self.experiment_progress.emit({"new_experiments": len(new), "total_experiments": self.conducted_experiments_count,
                                           "current_batch_count": self.conducted_experiments_count})
        summary_path = folder / "summary.json"
        if not summary_path.exists():
            return
        summary = self._read_campaign_json(summary_path)
        if summary is None:
            return
        if summary["status"] == "stopped":
            self.stop_execution()
            return
        if summary["status"] == "failed":
            raise RuntimeError(f"Campaign {summary['status']}; MODEL transition cancelled")
        if completed | skipped != self.expected_experiments or not summary.get("recording_complete"):
            return
        model_path = Path(self.controller.model_folder) / f"_model_{self.controller.omni_count}_{self.campaign_id[:12]}"
        # A lock while publishing campaign_input.json must allow this same
        # transition to retry without failing on the directory already created.
        model_path.mkdir(exist_ok=True)
        (model_path / "campaign_input.json").write_text(json.dumps({"campaign_id": self.campaign_id,
            "campaign_folder": str(folder), "experiments": sorted(completed),
            "skipped_experiments": sorted(skipped),
            "pbm_fitting": {"size_term": self.controller.pbm_size_term}}, indent=2), encoding="utf-8")
        self.controller.current_model_folder = str(model_path)
        self.controller.omni_count += 1
        self.omni_run_count = self.controller.omni_count
        self.begin_model_monitor(str(model_path))
        self.controller.current_stage = Stage.MODEL
        self.omni_completed.emit({"omni_run_count": self.omni_run_count, "model_path": str(model_path)})
        self.stage_changed.emit(Stage.MODEL)
    
    
    def begin_omni_monitor(
        self,
        path,
        num,
    ):
        """
        Begin monitoring a new OMNI batch.
        """
    
        self.previous_exp_count = 0
        self._seen_omni_files = set()
    
        self.set_monitor(
            mode=Stage.OMNI,
            path=path,
            num=num,
        )
    
    
    def begin_model_monitor(
        self,
        path,
    ):
        """
        Begin monitoring a MODEL run.
        """
    
        self.set_monitor(
            mode=Stage.MODEL,
            path=path,
            num=None,
        )
    
    
    def check_omni_transition(
        self,
        model_root,
    ):
        """
        Check whether the current OMNI batch is complete.
    
        If complete:
            OMNI -> MODEL
        """
    
        if self.controller is None:
            return None
    
        if self.controller.process_stop:
            return None
    
        self.controller.current_stage = Stage.OMNI
    
        if not self.controller.if_omni_finished(
            self.monitor_path,
            self.monitor_num,
        ):
            return None
    
        model_path = (
            self.controller.pathcreator_model(
                self.controller.omni_count,
                model_root,
            )
        )
        (Path(model_path) / "pbm_fitting.json").write_text(
            json.dumps({"size_term": self.controller.pbm_size_term}, indent=2), encoding="utf-8")
    
        self.controller.current_model_folder = (
            model_path
        )
    
        self.controller.omni_count += 1
    
        self.omni_run_count = (
            self.controller.omni_count
        )
    
        # Read completed OMNI results before moving to MODEL.
        if self.data_processor is not None:
            self.data_processor.read_data_from = (
                self.controller.current_omni_folder
            )
    
            self.data_processor.data_reader()
    
        self.begin_model_monitor(model_path)
        
        self.controller.current_stage = Stage.MODEL
        
        self.start_timer(3000)
        
        result = {
            "omni_run_count": self.omni_run_count,
            "model_path": model_path,
        }
        
        self.omni_completed.emit(result)
        self.stage_changed.emit(Stage.MODEL)
        
        return result
    
    
    def check_model_transition(
        self,
        omni_root,
    ):
        """
        Check whether MODEL is complete.
    
        If complete:
            MODEL -> next OMNI batch
        """
    
        if self.controller is None:
            return None
    
        if self.controller.process_stop:
            return None
    
        self.controller.current_stage = Stage.MODEL
    
        if not self.controller.if_model_finished(
            self.monitor_path
        ):
            return None
    
        self.controller.model_count += 1
    
        self.model_run_count = (
            self.controller.model_count
        )

        if self.campaign_mode:
            completed_model_path = self.monitor_path
            # MODEL has completed. Keep the loop active in the OMNI phase,
            # ready for an explicitly supplied next campaign. Never re-read
            # the previous campaign's completion markers or replay it silently.
            self.awaiting_campaign = True
            self.monitor_mode = Stage.OMNI
            self.monitor_path = None
            self.controller.current_stage = Stage.OMNI
            self.start_timer(500)
            result = {"model_run_count": self.model_run_count, "model_path": completed_model_path,
                      "awaiting_recommendation": True, "awaiting_campaign": True}
            self.model_completed.emit(result)
            self.stage_changed.emit(Stage.OMNI)
            self.next_experiment_required.emit(result)
            return result

        # Preserve the original external/manual loop when no optimizer adapter
        # is installed: the external producer supplies the next batch's files.
        creator = None
        if self.next_experiment_provider is not None:
            creator = self.next_experiment_provider(self.controller.current_model_folder)
            if creator is None or creator.actually_sampled_experiments < 1:
                raise ValueError("Next experiment provider returned no experiments")
    
        omni_path, exp_count = (
            self.controller.pathcreator_omni(
                self.controller.model_count,
                omni_root,
            )
        )
    
        self.controller.current_omni_folder = (
            omni_path
        )

        if creator is not None:
            creator.export_OMNI_expts(omni_path)
            exp_count = creator.actually_sampled_experiments
    
        self.begin_omni_monitor(
            path=omni_path,
            num=exp_count,
        )
        
        self.controller.current_stage = Stage.OMNI
        
        self.start_timer(3000)
        
        result = {
            "model_run_count": self.model_run_count,
            "omni_path": omni_path,
            "exp_count": exp_count,
            "external_experiment_required": creator is None,
        }
        
        self.model_completed.emit(result)
        self.stage_changed.emit(Stage.OMNI)
        
        return result

    def start_execution(
        self,
        omni_root,
        initial_exp_count,
    ):
        """
        Start the first OMNI execution cycle.
        """
    
        if self.controller is None:
            raise RuntimeError(
                "Controller has not been assigned."
            )
    
        if self.data_processor is None:
            raise RuntimeError(
                "Data processor has not been assigned."
            )
    
        if self.experiment_creator is None:
            raise RuntimeError(
                "Experiment creator has not been assigned."
            )
    
        # Allow execution.
        self.controller.process_stop = False
    
        # Create initial OMNI folder.
        omni_path, exp_count = (
            self.controller.pathcreator_omni(
                self.controller.model_count,
                omni_root,
            )
        )
    
        self.controller.current_omni_folder = (
            omni_path
        )
    
        # Initialize OMNI monitoring state.
        self.begin_omni_monitor(path=omni_path,num=initial_exp_count)
    
        # Export the experiment definitions.
        try:
            self.experiment_creator.export_OMNI_expts(omni_path)
        except Exception as error:
            self.handle_monitoring_error(error)
            raise
    
        # Start Controller execution.
        self.controller.start()
        
        # Announce the initial execution stage.
        self.stage_changed.emit(Stage.OMNI)
        
        # Start polling.
        self.start_timer(5000)
        
        return {
            "omni_path": omni_path,
            "exp_count": initial_exp_count,
        }
    
    
    def stop_execution(self):
    
        self.stop_timer()
    
        if self.controller is not None:
            was_active = (
                self.controller.current_stage is not None
            )
    
            self.controller.stop()
    
            if was_active:
                self.stage_changed.emit(None)
