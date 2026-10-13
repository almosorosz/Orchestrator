from PySide6.QtCore import QObject, QTimer, Signal

from utilities.data_utils import counter

from app.controller import Stage

class ExperimentCoordinator(QObject):

    omni_completed = Signal(dict)
    model_completed = Signal(dict)
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

        # ------------------------------------------------------
        # TIMER
        # ------------------------------------------------------

        self.timer = QTimer(self)

        self.timer.timeout.connect(self.monitoring_tick)


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
    
        Unexpected monitoring failures stop execution and
        are reported through error_occurred.
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
    
                current_count = counter(
                    self.monitor_path,
                    self.controller.omni_filename,
                )
    
                new_experiments = (
                    self.record_omni_file_count(
                        current_count
                    )
                )
    
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
    
    
    def begin_omni_monitor(
        self,
        path,
        num,
    ):
        """
        Begin monitoring a new OMNI batch.
        """
    
        self.previous_exp_count = 0
    
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
    
        omni_path, exp_count = (
            self.controller.pathcreator_omni(
                self.controller.model_count,
                omni_root,
            )
        )
    
        self.controller.current_omni_folder = (
            omni_path
        )
    
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
        self.experiment_creator.export_OMNI_expts(
            omni_path
        )
    
        # Start Controller execution.
        self.controller.start()
        
        # Announce the initial execution stage.
        self.stage_changed.emit(Stage.OMNI)
        
        # Start polling.
        self.start_timer(5000)
        
        return {
            "omni_path": omni_path,
            "exp_count": exp_count,
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