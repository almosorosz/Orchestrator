"""Exercise the frozen application's resources, recording, archive and presets."""
import json
from pathlib import Path
import time
import traceback


def run(output):
    window = None
    result = {}
    try:
        from PySide6.QtWidgets import QApplication
        from app.paths import DATA_DIR, WINDOWS_ICON_PATH, AS_PUMP_CALIBRATION_FILE
        from app.version import __version__
        from MAIN_WINDOW import MainWindow
        from experiment.models import Campaign, ExperimentDefinition
        from experiment.recording import RunRecorder
        from app.presets import write_preset, read_preset
        import pandas as pd
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        assert WINDOWS_ICON_PATH.is_file() and AS_PUMP_CALIBRATION_FILE.is_file()
        assert not window.windowIcon().isNull()
        window.btn_imed.click()
        assert window.stack.currentWidget() is window.imed_page
        assert not window.imed_page.panel.isWindow()
        assert window.imed_page.panel.import_button.isEnabled()
        assert not window.imed_page.panel.design.isEnabled()
        window.btn_dashboard.click()
        configs = {"pump": {"name": "Smoke-test pump", "type": "masterflex", "simulated": True, "connection": {}}}
        frame = pd.DataFrame({"time_s": [0, 1], "pump": [10, 20]})
        definition = ExperimentDefinition.from_frame(frame, configs)
        campaign = Campaign("Package smoke test", (definition,))
        recorder = RunRecorder(DATA_DIR / "smoke", campaign, configs)
        recorder.submit("run_start", "run1", {"experiment_id": definition.experiment_id, "planned": frame.to_dict(orient="list")})
        for i in range(2):
            recorder.submit("sample", "run1", {"elapsed_s": i, "pump.target": i + 10, "pump.speed_rpm": i + 10})
        recorder.submit("run_end", "run1", {"run_id": "run1", "experiment_id": definition.experiment_id, "status": "completed"})
        recorder.submit("summary", "", {"status": "completed"})
        recorder.close()
        assert len(pd.read_csv(recorder.folder / "run1" / "execution.csv")) == 2
        browser = window.plots_page.browser
        browser.root_input.setText(str(recorder.folder))
        browser.refresh()
        def wait(predicate):
            deadline = time.monotonic() + 20
            while not predicate() and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            assert predicate(), "Background operation timed out"
        wait(lambda: not browser._busy and browser.details and browser.details["folder"] == str(recorder.folder))
        browser.runs.selectRow(0)
        browser.plot_selected()
        wait(lambda: not browser._busy and not browser._image.isNull())
        setup = DATA_DIR / "smoke" / "setup.json"
        write_preset(setup, {"devices": configs, "profiles": {}, "experiments": [frame.to_dict(orient="split")]},
                     window.experiment_service.exp_par, (), {}, {"output_folder": str(DATA_DIR / "smoke"), "sql_enabled": False})
        read_preset(setup)
        result = {"passed": True, "version": __version__, "data_directory": str(DATA_DIR),
                  "resources": True, "imed_page": True, "csv_recording": True,
                  "campaign_plot": True, "setup_preset": True}
    except Exception:
        result = {"passed": False, "error": traceback.format_exc()}
    finally:
        if window is not None:
            window.close()
        Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result.get("passed") else 1
