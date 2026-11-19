# CrystalLoop code audit — 2026-11-19

Branch: `CrystalLoop_0.3`; starting commit: `e07d9cf`.

## Cleanup

- Removed 38 tracked, generated Python/notebook checkpoint copies and ignored future `.ipynb_checkpoints` directories.
- Removed the unreferenced `AppState` module and obsolete integrated BO VAE implementation. The documented standalone `bo_app` remains available.
- Removed unreferenced plotting, sampling, counting and prediction helpers, redundant attributes, unused local calculations, duplicate imports and unused imports.
- Preserved Qt callbacks, HTTP route handlers, hardware interfaces, notebook compatibility APIs, persisted configuration fields and references that retain Qt objects.
- Preserved original notebooks, calibration data, experiment campaigns, models and runtime stock records.
- Added a Ruff configuration and optional development dependencies for repeatable checks.

## Defects repaired

- Reservoir stock loaded from disk now rejects nonfinite, negative, boolean and nonnumeric amounts. In particular, `NaN` cannot bypass the stock comparison.
- An unreadable stock file leaves the application available, blocks tracked execution and requires confirmed physical amounts in Reservoir amounts. The unreadable file is retained until those amounts are confirmed.
- OMNI export now reads templates from the bundled resource directory, independently of `CRYSTALOOP_DATA_DIR` and the frozen application's writable data directory. Packaging includes the required templates.
- Sentinel page refresh identifies load completions by their originating URL. Late `about:blank` navigation and stopped-load events cannot complete a newer dashboard load.
- Controller basics are discoverable by the project's unittest runner. Older output-folder and sidebar assertions now check the current shared-folder and compact-layout behavior.

## Verification

- Final results across 46 test modules: **371 tests passed; zero failures, errors or skips**, after the complete run and affected-suite reruns.
- The complete automatic-loop suite passed all 15 tests, including BO/IMED DoE and Control Panel presets, PE/MD/DO follow-ups, recipe execution and cancellation. It took about ten minutes; the initial coverage-instrumented run exceeded the audit runner's per-module limit and was replaced by this completed run.
- Sentinel passed all 22 tests after the refresh fix, including the real dashboard served by a local test server. MSI's embedded graph test also passed.
- Ruff `F` and `E9`: clean. Vulture at 100% confidence: clean. Parsed all 204 remaining tracked Python files and the packaging specification; `git diff --check` passed.
- The audit runner's initial Windows multiprocessing bootstrap issue was corrected before rerunning the affected PE/design suites, which passed.

Lower-confidence unused-code findings were reviewed for dynamic callbacks, public interfaces and saved-file compatibility before deciding what to remove.

Tests use simulated devices, local HTTP servers and temporary files. They cover BO and IMED loops, PE/MD/DO and partial results, sensitivity and uncertainty, PBM numerical benchmarks, recipe generation, calibration, stock and mass tracking, recording/recovery, archive plots, MSI graphs, Sentinel desktop/history/thermal paths, and packaging resource lookup.

The audit does not certify physical hardware behavior. A new Windows executable or installer was not built as part of this source audit.
