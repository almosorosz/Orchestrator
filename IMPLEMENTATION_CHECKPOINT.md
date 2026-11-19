# CrystalLoop linked inputs implementation record — 2026-11-19

Branch: CrystalLoop_0.3. Base commit: a7bc4f5.
The implementation and targeted validation described below are complete.

## Accepted requirements

- Share MSI topology / material roles / measured quantities between Control Panel, DoE and loop assignment.
- Explain Automatic loop settings inline, use drop-down menus, and reuse loaded seed-pump calibration.
- Unseeded IMED: zero seed grams and initial D10/D50/D90 = 0/0/0.
- Optional average mixing: calculate ONE scalar from average RPM and average liquid volume over the selected experiment period. No additional RPM or volume profiles enter the PBM RHS. Unchecked means PowerRatio = 1.
- Per-experiment additions prefer AS scales; otherwise use calibrated pumping. Volumes use configured component densities (additive-volume estimate).
- Carry initial-campaign stirring/charge changes into IMED power-effect eligibility, BO decisions and SA inputs. Keep scientific parameter bounds from the candidate models; do not invent mechanisms/bounds.
- BO stage editor must expose Stirring as well as Temperature and Antisolvent.
- Define reservoir grams in Control Panel / Settings, show remaining quantities on dashboard, deduct delivered amounts, persist remaining stock, and block an understocked next experiment.
- Show estimated system mass in experiment window and execution.csv; opening the drain resets it, later additions accumulate again.

## Implementation locations

- `experiment/loop_assignments.py`: unique topology-based role/channel resolution, directed pump validation, applied DoE fallback and explicit overrides.
- `gui/device_connectivity.py`, `gui/system_topology_dialog.py`: scientific roles, measurement quantities and source-material labels alongside MSI topology.
- `experiment/material_balance.py`: calibrated pump integration and preparation-charge inference.
- `imed_python/mixing.py`: scalar average-RPM / average-volume calculation and provenance.
- `experiment/campaign_effects.py`: freeze initial operating-input eligibility; enable existing bounded power exponents only when power varies, disable unidentifiable exponent estimation when constant.
- `experiment/adaptive_loop.py`: per-experiment input generation, scale/pump fallback, process metadata, initial reactor mass and variable initial-charge execution.
- `bo/*`: stirring stage editing and scalar initial-volume decisions; initial-campaign eligibility gating.
- `imed_python/probabilistic*`: initial-volume SA multiplier and scalar power recomputation.
- `experiment/reservoirs.py`: thread-safe stock and estimated reactor-mass ledger; per-refill persistence.
- `gui/reservoir_dialog.py`: available grams, density and optional already-present reactor mass.
- `experiment_dashboard/experiment_dashboard.py`: pre-run stock checks, per-sample accounting, system-mass display/CSV and stock-error popup.

## Completed validation

- Final targeted batch: 25 tests passed, comprising 19 linked-input tests, a manual BO loop integration, and five non-WebEngine topology/control-panel tests.
- Linked-input coverage includes topology inference/ambiguity/direction, seed calibration reuse, unseeded state, scalar mixing averages, initial charges and volume execution, power-term eligibility, stirring stages, old BO save migration, portable command profiles, stock accounting/persistence, peak stock draw before pump returns, scale correction/drain reset, scale-free pump import, and stopping before an understocked second run.
- The MSI/stock integration executes three starting experiments plus a BO follow-up, verifies automatic assignments, deducts reservoir stock from 100 to 80 g, and checks system_mass_g in all four execution CSVs.
- Release integration batch: 31 tests passed, including the linked-input suite at that point, external-model guidance, experiment definition, a five-run IMED preset, and used-values preview.
- Earlier integration batch: 36 tests passed, including loop setup, pump calibration, run stages, a BO follow-up, and IMED one-off recipe execution. These batches overlap; counts are not additive.
- BO regression suite: four tests passed. IMED SA/profile suite: 17 tests passed.
- The supplied campaign_6ab362bda4994892bfa1a2c88cc26d60_d72354da was imported through the real BO campaign-import dialog: all five stirring profiles were recovered. Their modeling-period mean RPM values are approximately 322.66, 209.05, 201.45, 205.00 and 391.98.
- Visually reviewed average mixing, MSI assignment, reservoir contents/dashboard, and the actual campaign's Stirring stage editor. Reservoir headers fit without clipping.
- git diff --check passed. No physical hardware was actuated during validation.

## Validation limits and preserved user data

- An offscreen embedded WebEngine graph test stalled and was interrupted. The independent topology/control-panel tests passed; this is not a completed validation of graph rendering in the user's desktop session.
- Temporary-directory permission failures in earlier sandboxed tests were resolved by rerunning those tests with the required access. New linked tests prefer workspace temporary directories.
- Unrelated user runtime logs, campaigns, model folders, stock state and pre-existing temporary folders are excluded from the source commit and preserved in place.

## Scientific limits to retain

Mean RPM and mean volume give a scalar specific power input in m²/s³ through the existing PowerRatio slot. This affects secondary nucleation and agglomeration in the current PBMs. It does not add primary-nucleation dependence or establish causality by itself. Separate RPM and volume effects need independently varied DoE conditions. Initial charge volume is an executable BO decision only with defined preparation charging pumps; AS-driven volume changes remain controlled through AS profiles.
