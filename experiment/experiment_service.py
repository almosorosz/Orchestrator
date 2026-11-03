from app.paths import (
    AS_PUMP_CALIBRATION_FILE,
    SOLUTION_PUMP_CALIBRATION_FILE,
    SOLVENT_PUMP_CALIBRATION_FILE,
)

from experiment.pump_calibration import PumpRateConverter

from experiment.doe.doe_sampler import DoESampler
from experiment.doe.doe_creator import ExperimentCreator

from experiment.profiles.profile_gen import (
    ExperimentProfileParameters,
    ExperimentProfileStorage,
    ExpComponentGen,
    ProfileGenerator,
)
class ExperimentService:

    def __init__(self):

        # ------------------------------------------------------
        # PUMP CALIBRATION
        # ------------------------------------------------------

        self.as_rate_converter = PumpRateConverter(
            str(AS_PUMP_CALIBRATION_FILE),
            "File  W: AS ",
            "File R: Mass ",
        )

        self.solution_rate_converter = PumpRateConverter(
            str(SOLUTION_PUMP_CALIBRATION_FILE),
            "P01:Value (Setpt)",
            "File R: Mass ",
        )

        self.solvent_rate_converter = PumpRateConverter(
            str(SOLVENT_PUMP_CALIBRATION_FILE),
            "P01:Value (Setpt)",
            "File R: Mass ",
        )

        self.cleaning_sol_rate_converter = PumpRateConverter(
            str(SOLVENT_PUMP_CALIBRATION_FILE),
            "P01:Value (Setpt)",
            "File R: Mass ",
        )

        # ------------------------------------------------------
        # EXPERIMENT PROFILE
        # ------------------------------------------------------

        self.exp_par = ExperimentProfileParameters()

        self._build_doe_dependencies()


    def _build_doe_dependencies(self):
        """
        Recreate DoE helper objects while preserving the current
        experiment-profile parameters.
        """

        self.doe_sampler = DoESampler()

        self.component_gen = ExpComponentGen()

        self.profiles = ProfileGenerator(
            self.exp_par,
            self.component_gen,
        )

        self.exp_storage = ExperimentProfileStorage(
            self.as_rate_converter,
            self.solution_rate_converter,
            self.solvent_rate_converter,
            self.cleaning_sol_rate_converter,
        )


    def build_creator(self):
        """
        Create a fresh ExperimentCreator using the current
        experiment-profile configuration.
        """

        return ExperimentCreator(
            self.doe_sampler,
            self.component_gen,
            self.exp_par,
            self.profiles,
            self.exp_storage,
        )


    def generate_initial_doe(
        self,
        initial_exp_count,
        strategy_code,
    ):
        """
        Rebuild temporary DoE dependencies and generate the
        requested initial design.
        """

        self._build_doe_dependencies()

        creator = self.build_creator()

        creator.set_doe_settings(
            self.exp_par,
            initial_exp_count,
            Type=strategy_code,
        )

        creator.generate_experiments()

        return creator


    def reset(self):
        """
        Restore experiment/profile services to their initial state.
        """

        self.exp_par = ExperimentProfileParameters()

        self._build_doe_dependencies()