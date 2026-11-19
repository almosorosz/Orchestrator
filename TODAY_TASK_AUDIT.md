# CrystalLoop: 16-point implementation audit

Date: 2026-11-19. Branch: `CrystalLoop_0.3`. Status: all 16 items implemented and checked.

This list reconciles today's accepted requests with the implementation and regression
coverage. Earlier experiment-definition requests are included under experiment design
and preview; later recording and setup fixes are included under their respective items.

| # | Requirement | Implementation and verification |
|---|---|---|
| 1 | Clearer Settings with every existing capability retained | Four full-width workflow tabs replace the narrow sidebar. Folder fields expand, errors wrap, and original controls remain connected. `gui/settings_workflow.py`; `test_setup_workflow`. The experiment-definition editor retains its compact screen-fitting layout, wider timeline spacing, and shared/independent device profiles. |
| 2 | Understandable experiment definitions and DoE previews | Readable stage/control names and units, defined-control table, default per-experiment profile preview, factor separation statistics and selectable 3D axes. Single experiments without sampled factors also display. Equal-to-previous is available for all device targets; shorter profiles retain their final value. Unsaved representative previews, consistent time/rate units, stage-aware validation messages and Control Panel profile viewing are retained. `test_setup_workflow`, `test_recipe_preview`, `test_recipe_editor`, `test_recipe_time_units`, `test_profile_synchronization`. |
| 3 | Automatic loop checklist with navigation and live green states | Checks batch, folders, experiment period, devices/monitors, channels, process inputs, BO objectives or IMED models, PE responses, design bounds and stopping rule. Optional advanced inputs remain identifiable. Clickable items navigate; draft settings survive navigation until Save/Cancel. `gui/setup_checklist.py`, `gui/setup_readiness.py`, `gui/loop_setup_dialog.py`; `test_setup_workflow`, `test_loop_setup_dialog`. |
| 4 | IMED checklist showing what needs filling | Experimental data, corresponding process records, solubility, candidate parameter definitions and PE responses/settings are checked. Fitted results are displayed as a later optional step. Every original modeling action remains accessible in the three workflow tabs. `imed_python/control_panel.py`; `test_setup_workflow`. |
| 5 | Reuse MSI/Control Panel/DoE assignments and explain loop fields | Directed MSI roles, applied DoE assignments and measurement labels resolve existing devices/channels; ambiguity needs an explicit selection. Recording channels and device assignments use menus. Used values shows effective values and provenance; explanatory text accompanies overrides, support profiles, calibrated delivery and telemetry. `experiment/loop_assignments.py`, `gui/loop_setup_preview.py`; `test_linked_inputs`, `test_loop_setup_dialog`, `test_adaptive_loop`. |
| 6 | Reuse seed calibration and support unseeded IMED | Loaded pump calibration is reused. Zero seed mass with 0/0/0 initial seed percentiles produces no initial seed population. Positive seed mass requires usable percentiles. `experiment/pump_calibration_files.py`, `experiment/adaptive_loop.py`, `imed_python/campaign_import.py`; `test_linked_inputs`. |
| 7 | Optional average RPM/average volume mixing input | One scalar per experiment: `Np × (average_RPM/60)^3 × diameter_m^5 / average_volume_m3`. Disabled means `PowerRatio = 1`. Scale-derived addition is preferred, with calibrated pumping as fallback; volumes use configured densities. No RPM or volume profile is added to the PBM state. `imed_python/mixing.py`, `experiment/adaptive_loop.py`; `test_linked_inputs`, `imed_python.test_mixing`. |
| 8 | Dynamic operating inputs in IMED, BO and SA; editable BO stirring | Initial-campaign variation controls operating-input eligibility. IMED uses existing bounded secondary-nucleation/agglomeration power terms, BO imports and edits stirring stages, and SA recomputes scalar power when RPM/volume change. Existing disabled mechanisms and fixed scientific bounds are preserved. `experiment/campaign_effects.py`, `bo/profiles.py`, `bo/stage_editor.py`, `imed_python/probabilistic.py`; `test_linked_inputs`, `test_bo_controls`, `imed_python.test_mixing`. The supplied five-run campaign was also imported through the actual BO dialog during the linked-input implementation. |
| 9 | Reservoir quantities, dashboard display and stock checks | Define available grams in Control Panel/Settings, display configured stock on Dashboard, deduct delivered amounts, persist across sessions and reject an understocked next experiment. Invalid or unreadable stock cannot bypass checks. Undefined reservoirs remain untracked. `experiment/reservoirs.py`, `gui/reservoir_dialog.py`; `test_linked_inputs`. |
| 10 | Reactor/system mass display and recording | Charge, seed and AS additions accumulate estimated mass; opening the discharge valve resets it; later additions accumulate again. `system_mass_g` is displayed and written to execution CSVs. `experiment/reservoirs.py`, `experiment_dashboard/experiment_dashboard.py`; `test_linked_inputs`. |
| 11 | Complete BO and IMED loops from either starting source | Control Panel and Settings DoE presets execute, collect data, populate model inputs, calculate recommendations and execute follow-ups. IMED refits retained candidates after each new observation and branches to MD or DO. Configured follow-up limits and Stop control termination. Stage purposes and the selected continuous experiment period preserve preparation/cleanup while restricting modeling data. `experiment/adaptive_loop.py`, `experiment/run_stages.py`; `test_adaptive_loop`, `test_run_stages`, `test_loop_recording`. |
| 12 | Explain and save every modeling stage | Logs show imports, calculations, selection decisions and launches. Named BO/IMED stage history preserves evaluated/retained models and training data. MODEL folders contain input CSVs, fitted models, parameters/covariances and loadable PE/stage files; final observations are saved/refitted. `imed_python/session.py`, `experiment/adaptive_loop.py`; `test_adaptive_loop`. |
| 13 | Semi-automatic recipe execution and retained results after Stop | Execute-recipe actions validate/reuse assignments, run once, then import recordings into the modeling dataset. Stopped PE, MD, DO, reliability and sensitivity calculations retain completed usable results. Completed run indicators remain green. BO control sets have no fixed four-variable cap; fitting and selection flags round-trip. `test_adaptive_loop`, `test_partial_results`, `test_bo_controls`, `test_release_a`. |
| 14 | Shared general Exp/Model output locations | Settings destinations are reused by Control Panel, independent DoE, Dashboard automatic loop and recipe execution. Both original CSV/external loop and native BO/IMED remain available. External MODEL info explains current file handoff and exports an actual experiment template plus an AI assistance prompt; no planned features are presented as an interface. `test_external_model_info`, `test_adaptive_loop`, `test_release_a`. |
| 15 | Repair recent failures and preserve compatibility | NumPy boolean/scalar serialization is normalized in recordings and stage events. Monitor inclusion/restoration, OMNI bundled resources, Sentinel load completion, archive multi-campaign/device selection, old BO stirring saves and stock persistence are covered. Generated source checkpoints and confirmed unused code were removed in the preceding audit. `test_recording`, `test_run_stages`, `test_external_model_info`, `test_archive`, `test_linked_inputs`; earlier full audit in `CODE_AUDIT.md`. |
| 16 | Validate scalar epsilon in nucleation/agglomeration and numerical integration | Analytical unit/scaling/averaging tests, independent PBM reference comparison, mesh refinement, mass conservation and SA recomputation pass. The step-command averaging and stale SA initial-volume defects found during validation were repaired; invalid zero-power/negative-exponent combinations now give a clear error. `imed_python/test_mixing.py`, `imed_python/mixing.py`, `imed_python/probabilistic.py`, `imed_python/solver.py`. |

## Numerical evidence for point 16

- With `Np = 2`, diameter `0.04 m`, average speed `300 RPM` and volume `100 mL`, epsilon is `0.256 m²/s³`. Doubling RPM multiplies it by eight; doubling volume halves it. Disabling mixing returns one.
- Time weighting distinguishes held commands from ramps: a 100 RPM hold for five seconds followed by a 300 RPM hold for five seconds averages 200 RPM, while the corresponding linear ramp/hold averages 250 RPM.
- An analytical ramped antisolvent addition matches the integrated mean-volume result; a 101-point sampled scale trace changes epsilon by less than `2e-7` relative to that reference in the test case.
- Secondary nucleation alone, agglomeration alone, and their combination agree with an equivalent exact kinetic-prefactor rescaling to within `1e-6` relative to each output's peak magnitude using tightened solver tolerances.
- An independent QMOM comparison gives maximum normalized count errors of approximately `0.01696%`, `0.00744%` and `0.00256%` for 80, 160 and 320 finite-volume cells. Relative API mass conservation errors are about `1e-15` in this benchmark; states remain finite and nonnegative observables are retained.
- SA varying RPM and initial volume by a factor of two produces the expected fourfold epsilon. The PBM state still contains its original `N + 6` components.

The current equations couple epsilon to **secondary nucleation and agglomeration**.
Primary nucleation has no epsilon term and remains unchanged; the regression explicitly
checks this. A zero kinetic power exponent gives no mixing dependence. Estimating an
effect requires enabled mechanisms, nondegenerate exponent bounds and informative data.

These tests validate numerical incorporation for the benchmark cases, rather than
physical calibration for an arbitrary vessel. Charged volume is an additive-volume
estimate using configured densities. The chosen scalar uses average RPM and average
volume, not the time-average of instantaneous power. Separate effects need informative,
independently varied experimental conditions.

## Verification and delivery

- The final broad regression batch passed **204 tests** across guided setup, recipe generation/preview, profile synchronization, stages, external handoff, linked inputs, stock/mass, partial results, BO controls, archive, recording, loop handoff and PBM/design/SA numerics.
- The complete automatic-loop suite passed all **15 integration tests** in 638 seconds: five-run BO and IMED from both Settings DoE and Control Panel, PE/MD/DO branching and stage files, recipe execution/import, statistical screening, preflight and cancellation.
- After the last display changes, all **15 guided-setup/loop-dialog tests** passed, including legacy presets, inherited-target previews, preserved sampling statistics, draft navigation, refreshed device menus and green-state updates. This rerun overlaps the broad batch.
- The average-mixing module passed all **9 numerical tests** and is also included in the broad batch.
- Settings design/preview, Automatic loop and IMED were visually reviewed using offscreen Qt renders.
- Ruff checks passed, all 209 Python source files parsed, and `git diff --check` passed.

Tests use simulated hardware and local temporary files. Existing user campaigns,
models, inventory and logs are preserved. The source implementation is delivered on
`CrystalLoop_0.3`; an executable/installer is not rebuilt by these source changes.
Restart the source app to see the new windows.
