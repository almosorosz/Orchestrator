# CrystaLoop 0.4.0 — Campaign archive, recovery, presets and Windows release

## Guided experiment and automatic loop setup

Settings now has four full-width tabs: **Experiment design**, **Preview & devices**,
**Automatic loop**, and **Files & presets**. The checklist above them shows missing
requirements in amber and configured items in green. Click an item, or **Review next
item**, to reach its settings. Generate the experiments, inspect individual profiles,
then validate and apply device assignments. Sampling diagnostics retain minimum and
mean pairwise factor separations and let you select the three axes of the 3D plot.

**Automatic loop setup** has its own live checklist for the starting batch, output
folders, experiment period, devices, recording, process inputs and modeling settings.
For IMED it also checks PE responses and follow-up design bounds. Optional overrides,
support profiles and PBM simulation controls are under **Advanced**. **Used values**
shows inherited values and their sources. You can open Control Panel or the modeling
page and return to the same draft; **Save** applies it and **Cancel** discards it.
Dashboard **Start** still performs execution preflight and selects the starting batch
if both Control Panel and Settings experiments are available.

IMED also shows a clickable checklist above **Case setup**, with the modeling actions
grouped into **Inputs & models**, **Fit & design**, and **Analysis & history**.
Fitted results are a subsequent step rather than a prerequisite for starting PE.

The [16-point implementation audit](TODAY_TASK_AUDIT.md) records the completed scope,
regression evidence and numerical validation of average mixing inputs.

## IMED

Open **IMED** in the sidebar to import a MATLAB model or a recorded CrystaLoop
campaign. The page provides forward PBM simulation, parameter estimation, model
discrimination, and D-optimal design. To use campaign data for modeling, load a
MATLAB case with the relevant kinetics and solubility before importing the campaign.
See the [IMED workflow guide](imed_python/README.md) for input and solver details.
For PBM fitting, IMED Parameter Estimation offers final D10/D50/D90 or all recorded size trends under **Size samples**. **Settings → PBM size samples in LOOP** chooses the same data mode for the automated EXP–MODEL handoff and is retained in setup presets. Campaign MODEL inputs carry the choice in `campaign_input.json` under `pbm_fitting.size_term`. The CSV-based external loop writes `pbm_fitting.json` in its MODEL folder. External MODEL scripts should read this value when constructing their fitting objective. The Python fitter accepts it as the `terms` argument to `FitObjective`.
JAX/Diffrax is optional when running from source and is not bundled in the
Windows release; use SciPy PBM in the packaged app.

## Recorded campaign plots

### Fast simulated profiles

Accelerated simulated experiments use event-aware simulation time: every profile
waypoint is sampled, with at least eight intervals along changing ramps. Device
responses and simulated delivery use that same clock, so short additions are not
skipped when wall-clock processing runs late. The requested speed (for example
100×) is a target; detailed short segments can take longer under processing load.
Real and mixed-device runs retain their real-time execution timing.

Live readings are delivered to the GUI in batches. Plot redraws remain throttled
and off-thread, while the samples between redraws are retained. Bounded display
history preserves profile boundaries and extrema across the experiment instead
of discarding its oldest section. Full-resolution samples remain in the CSV/SQL
recordings. Very short additions can still occupy few screen pixels on a long
time axis; the recorded samples remain available in Data & Plots.

New manual and sampled campaign batches use ordered run IDs and matching folders,
for example `1_<unique-id>`, `2_<unique-id>`, `3_<unique-id>`. Numbering restarts at
one for each campaign execution. Run metadata includes the experiment number and
batch name; existing unprefixed campaign records remain readable.

Open **Data & Plots → Recorded campaigns**. It discovers Settings recordings,
Control Panel recordings, the LOOP experiment folder, and previously used recording
roots. Browse can select another root or a single campaign. The list refreshes
every three seconds while visible; **Refresh list while running** can disable this.
Use Ctrl-click to select several campaigns or Shift-click for a range. The
experiment table combines their runs and shows the source campaign for each row.
Experiment status shows queued, running, saving, completed,
interrupted, or not executed. Completed experiments can be inspected while the
remaining experiments run or wait in the queue.

Ctrl/Shift-select runs and choose **Plot selected** to compare target and measured
values against each experiment's elapsed seconds.
With no experiment rows selected, **Plot selected** uses every executed run in
the selected campaigns.
Selected experiments overlay on shared device panels, with a distinct color for
each experiment and line styles distinguishing target/measurement fields. Compact
controls and short selection tables leave most of the page for the graphs.
Open **All devices** to check any combination of device plots. **Fit width** and
zoom control the cached plot.
Enter elapsed seconds in **From (s)** and **To (s)** before plotting to limit the
time window. Leave either bound blank to keep that end open. **Clear plots**
clears the display and discards pending plot results.
Each click reads a snapshot; plotting a running experiment does not follow it live.
Read/plot/export work runs off the GUI thread, separately from acquisition. Display
traces retain up to 4,000 points using extrema-preserving reduction; CSV recordings
and exported files retain every saved sample. Torque and valve commands use a
separate axis where their units differ. **Export CSVs** copies fully saved,
completed experiments to a new directory, grouped by campaign for a multi-campaign
export. **Open folder** opens the current campaign's recording folder.

## Recording recovery

The selected recording output root contains `application.log`, including dashboard
messages and execution logs. Starting a manual, independent DoE, or LOOP campaign
switches logging to that run's selected output root. Logs append across sessions
and rotate at 5 MB with three backups. The internal diagnostic log under
`data/runtime/application.log` is retained as a fallback. Output log files are
closed between messages to reduce conflicts with folder synchronization.

Startup checks known recording roots in the background. OS-owned writer locks
protect campaigns still owned by a running process. Interrupted recordings replay
their durable pending samples from the last committed CSV position. Truncated final
sample fragments are retained as `.partial` files; complete rows are preserved.
Completed experiment statuses remain completed. Unfinished runs become interrupted;
queued experiments are never started, and hardware/LOOP execution does not resume
automatically. SQL outbox replay is idempotent and only applies when SQL was enabled.

If Excel/OneDrive locks a file, close it and click **Recover interrupted recordings**
to retry. Older recordings without a writer ownership lock are skipped automatically.
Close all older app instances, then enable **Include legacy recordings in recovery**
to recover those explicitly. Recovery locks protect local processes; do not recover
a cloud-synchronized campaign that is executing on another computer.

## Reusable setup presets

Use **Settings → Save setup preset / Load setup preset**. A versioned JSON saves the
device list, simulation settings, limits, editable/manual profiles, manual recording
folder, LOOP folders, sampling settings, process parameters, profile assignments,
and recording/optional SQL settings. The exact generated DoE conditions and profile
tables are included when available, together with unit operations and connectivity.
Presets can include SQL connection URLs with credentials. Loading requires an idle
session. Standard presets restore disconnected devices. The bundled
[five-run simulation presets](examples/configurations/README.md#five-run-simulation-presets)
prepare their simulated devices and assignments on load. A saved generated DoE restores without sampling again,
including random samples. Open **Assign devices** to review the saved selections
against the current devices, then preview and apply the assignments. Connect the
required devices before starting. Valid saved LOOP folders are restored automatically.
Presets without generated data still require DoE generation.

## Bayesian optimization

Install the BO runtime in the Python environment running CrystalLoop.

```powershell
python -m pip install -r requirements-bo.txt
python -m bo
```

The same workflow is available under **IMED → BO** in CrystalLoop. Open an
IMED MATLAB or CrystalLoop case, import a recorded campaign, or reuse the current
IMED case. Select completed experiments, set input bounds, enable the targets,
then configure optional output limits under **Output screening**. Give each
target a scale in its own unit so that weights compare dimensionless losses.

**Physical profiles** starts with temperature and antisolvent stage lists inferred
from the selected experiments. Add, delete, or reorder stages, then set durations
and endpoints in seconds, °C, or g/s. Equal temperature endpoints define a hold.
Antisolvent stages can turn the pump off, hold a positive rate, or ramp the rate.
Select as many decision controls as your recipe needs. Add stages to expose more
durations, temperatures, or feed rates. Grouped controls include start delay,
final temperature, and total antisolvent dose. The selected runs still need to
show variation for BO to search along a control.

BO fits the selected physical controls to the recorded profiles and estimates
the proposed recipe from those fits. Compare recorded and fitted traces before
searching. In **Compare profiles**, pick a reference experiment and use **Use
trace as template** to preserve its transitions. **Compare all selected experiments**
overlays each trace and fit and shows the RMSE for each run. The **Decision controls**
tab shows fitted values for the reference experiment. Controls with the same fitted
value across the selected data are marked **Fixed**. Leave **Batch horizon (s)**
blank to use the longest selected experiment, or enter a longer horizon to fit
an extended recipe.
Temperature holds at the final setpoint after its last stage and antisolvent
feed stops after its last stage. Duration bounds apply to the fixed horizon.
Seeded proposals add seed at time zero, and unseeded proposals use zero seed mass.

**Propose experiment** shows the physical recipe, input history, and one-dimensional
GP projections through the proposal. **Observed inputs** shows standardized
values together or any input in its own units. **GP projections** shows the
predicted mean, a predictive band at ±1.96 standard deviations, the acquisition
for the weighted objective, and the actual measured points when available. The
output filter uses the selected margin and requires the full predicted interval
to meet every enabled bound. With small datasets, use the plots and reconstruction
errors to judge whether the proposed recipe is suitable for experimental review.

**Save BO setup** includes the selected runs, settings, stage templates, and
chosen physical controls. Older setups migrate their data and settings to
physical stages. **Save proposed recipe** exports the stages, plotted profiles,
and predictions as JSON for review. Install scikit-learn in the Windows build
environment to include the BO optimizer.

In IMED Parameter Estimation, **Optimization progress** shows the live
generation plot beside the optimization log. Drag the divider to resize them.

## Unit operations and MSI connectivity

In **Control Panel → Unit operations / modules**, define module codes/names and
unit operation codes/names/types (free text, such as crystallizer or container).
Apply the definitions, then use each device card's connectivity fields. Pumps use
source and target unit operations: positive/clockwise flow is source → pump → target.
Other devices attach to one unit operation. Their module must match that operation;
a pump has its own owning module and may draw from or deliver to containers in
any module. Choose **Pump module**, **From unit**, and **To unit** independently;
container dropdowns display their module. Changing the source preserves an
already selected pump module. Inter-module links are generated automatically in MSI.

Control Panel device cards use compact fields and two rows of action buttons.
The number of columns adapts to the available width, and toolbars wrap on narrow
windows. Resizing preserves the existing device instances and configuration.

Select controlled, read-only, or excluded roles. Read-only Huber devices are monitored
without starting, stopping, or setting their temperature. Existing devices default to
excluded from the physical graph until assigned; existing control behavior continues.
Physical connectivity is separate from assigning generated experiment profiles.

**Generate MSI / connectivity graph** validates the included devices and shows MSI
text and an offline interactive PyVis graph, using the exact renderer and style of
`notebooks/MSI_graph_compatible.py`. Unit operations including pumps and stirrers
are squares colored by module, controllers are yellow circles, read-only probes
are red circles, and module sensors are module-colored triangles. Each pump has
a flow-control node. Each controlled JKem
stirrer has a stir-rate control node and a read-only torque probe; passive stirrers
have read-only speed and torque probes. Stirrer MSI entries use `U_S_<device ID>`,
`C_SR_<device ID>` and `R_TQ_<device ID>`. Their outgoing connection identifies the
vessel they operate in and is shown grey/dotted, separately from material flow.
Directional flow/measurement links are solid; control links are dashed. Equipment,
module sensor groups and flow arrows share their module's color. A color legend
identifies each module; unit-attached read-only probes remain red. Edges use their
source module color, except grey stirrer-placement connections.
**Auto layout** re-runs network stabilization; **Show labels**
toggles MSI identifiers. Names and module membership appear in node tooltips instead of rigid
module boxes. The notebook's MSI grammar is retained; codes are escaped
to support arbitrary identifiers. The parser handles multiple module connections,
isolated units and cycles. Move nodes, zoom/pan, highlight neighborhoods by clicking,
copy/save MSI, or export the visible graph as PNG. The
notebook's old renderer still needs its own fixes for arbitrary identifier codes;
use the application's renderer for these definitions. Draft definitions can be saved
before every device is assigned. Control Panel configurations and setup presets
retain the definitions and device connectivity; campaign recordings include them.

**Control Panel → Module sensors…**, beside Pump calibration files, opens a table
of sensor code, name and module code. Enter codes without the `S_` prefix and use
unique codes across the system. Apply definitions, then generate MSI. For module
`m1` and sensor `airT`, the generated section includes `S_m1[[S_airT]]`. Sensors
appear as module-colored triangles and are retained in configurations and setup
presets. These are system-definition entries; they do not create acquisition
channels or hardware connections. Configure the corresponding read-only devices
separately when measurements are required.

The MSI dialog prioritizes the graph viewport and defaults to a landscape layout.
Enable **Show MSI code** to reveal the text; copying/saving works while it is hidden.
Data & Plots uses shorter selection tables and compact controls. **Hide lists**
reclaims more plotting space without clearing selected experiments.

`notebooks/MSI_graph_compatible.py` provides standalone notebook-compatible encoder
classes (dependencies: networkx, matplotlib, pyvis). The desktop embeds the same
HTML/JavaScript graph in Qt WebEngine; resources are bundled without CDN requests.
Run the file after
any older class definitions. Notebook graphs use separate iframe documents and
unique container IDs, so successive cell outputs remain independent. Use
`display(cc.modular_system_graph())` explicitly when a graph call is not the last
expression in a cell; use different filenames to retain multiple exported HTMLs.

## Windows installation and builds

The Windows x64 release ZIP contains a portable `CrystaLoop/CrystaLoop.exe` and
**Install.cmd**. Extract the complete ZIP and run Install.cmd for a per-user install,
desktop/Start menu shortcuts, and an uninstall entry; Python/admin rights are not
required. Uninstall retains recordings and settings. Frozen builds store writable
data in `%LOCALAPPDATA%/CrystaLoop/data`; bundled calibration/assets are read-only
resources. Source-checkout data stays in the repository's `data` folder. The optional
`CRYSTALOOP_DATA_DIR` environment variable overrides the writable data location.

Build on Windows with `python -m pip install --target .build-deps pyinstaller==6.16.0`,
then `python -m utilities.build_windows_release`. The ZIP is written to `release/`.
The executable carries version information and the embedded app icon. The build is
unsigned. `BUILD_INFO.json` reports optional NI-DAQ/PostgreSQL modules included;
vendor drivers still need installation on the device PC. Third-party license notices
are included. No personal recording folders, runtime settings or SQL URLs are bundled.
Startup failures are written to the writable data directory's
`runtime/STARTUP_FAILED.txt`. The build checks bundled project bytecode against
current source and includes the DoE library's required orthogonal-array tables.

Python/PySide6 laboratory experiment application. The current entry point is
`MAIN_WINDOW.py`. Release A adds hardware-free simulation, validated experiment
definitions, explicit execution outcomes, regression tests, and reliability fixes.
Release B adds device assignment and execution of generated DoE campaigns plus
CSV exports. The deliberate real Huber `start()` behavior remains unchanged.

## Generate, assign and execute a DoE

1. Configure/connect devices in **Control Panel** (or load the simulated demo and
   Connect All). Choose **Simulation speed** explicitly; real hardware always uses 1×.
2. Open **Settings**, enter the experiment count, edit process conditions and select
   a sampling strategy, then **Generate experiments**. OMNI/MODEL folders are not
   required for local generation/execution.
3. Select **Assign devices / preview DoE** under Environmental variables. Active roles are marked `*`.
   Assign each role to a distinct compatible device; inactive zero-only roles may
   remain unassigned. Device IDs, real/simulated mode and connection status are shown.
4. Ismatec uses the fixed **1 g/mL** assumption: generated **g/hour** becomes
   **mL/min** via `g/h ÷ 60`; no density entry is required. Masterflex
   profiles use the existing named role calibration to convert g/hour to RPM.
   Seed pumping is already RPM and supports Masterflex; no implicit RPM-to-flow
   conversion is made. Temperature and stirrer targets retain °C and RPM.
5. Select **Preview / validate**. Review experiment durations and target ranges.
   Source hours become seconds; return-pump segments from the existing generator
   are retained. Exact duplicate boundary rows are coalesced; conflicting targets
   at an identical time are rejected. Defaults can describe many hours of work;
   generation does not shorten experiments automatically.
6. **Apply assignments** saves the validated mapping without starting devices.
   Connect the assigned devices, then select **Start DoE independent from loop**
   in Settings. This opens the existing Experiment dashboard and executes the
   generated profiles in order. Only assigned actuators and selected connected
   read-only monitors are included. Unrelated devices are never started by this
   campaign. The same hardware ownership and Stop/End Session guards apply.

An assignment change invalidates the preview. All assigned devices must be
connected at launch; limits are revalidated before hardware is reserved. Assignment
presets can be saved/loaded as JSON. Missing/incompatible devices in a preset are
reported. Use the matching instrument calibration before physical execution.

## CSV exports

### Pump calibration files

Open **Control Panel → Pump calibration files…**, or **Pump calibrations…** in
the DoE assignment dialog. Select a pump and **Load CSV…**. Choose the header row,
file format and column mapping, then **Preview / validate → Apply calibration**.

Flow/target tables use measured flow in **g/h** and pump targets in **RPM** for
Masterflex or **mL/min** for Ismatec. For example:

```csv
flow_g_h,rpm
0,0
60,10
120,20
```

Use your measured values. Time/target/mass logs also work: select a time column,
its unit (seconds/minutes/hours), a pump-target column and a mass column in grams.
The loader estimates flow from the mass slope of each constant-target segment.
Loaded points are interpolated linearly; targets above the calibrated flow range
are rejected rather than extrapolated. Device target limits are still enforced.

The calibration points, source filename and checksum are saved with the device
configuration and setup preset. The original CSV is not required after loading.
Custom calibrations apply to assigned DoE mass-flow profiles. Manual profiles and
seed profiles already expressed in device units are not converted again. Changing
a calibration after assignment requires previewing/applying assignments again.
**Use built-in conversion** restores the existing Masterflex role calibration or
the Ismatec fixed 1 g/mL conversion. Only stirrers record and plot torque; pump
torque fields in older recordings are omitted from campaign plots.

**Export generated DoE CSV** in Settings creates a new `doe_<id>` subfolder with
`parameters.csv`, `generated_profiles.csv` and `metadata.json`. Profiles are wide:
one `time_s` column followed by the profile/device columns side by side. Separate
`expN_generated.csv` and `expN_assigned.csv` files omit campaign/experiment grouping
columns. Combined files include experiment identifiers so their timelines stay distinct.
Parameters include
per-experiment sampled values and fixed scalar settings. Units and conversion
details are stored in assignments/metadata.

After a successful assignment preview, **Export campaign CSV** additionally writes
`assigned_profiles.csv` and `assignments.csv`, including device IDs, converted
targets/units, interpolation and stable experiment/campaign IDs. Metadata records
conversion details, calibration coefficients and device configuration. Exports
never overwrite earlier export folders. These are **planned profiles**.

## Continuous recording and SQL (Release C)

Open **Settings → Recording / SQL settings**. Local recording is
automatic for every experiment. Choose the shared **Experiment output folder**
in Settings; an unset location uses the existing recording default (`data/runs`
on a new installation).
SQL is independently optional: uncheck **Also save to SQL** for CSV/JSON-only
recording. New installations default to SQL off. Enabling SQL creates the local
database; leaving Remote SQL URL blank keeps it local-only. Turning SQL off stops
future uploads but does not delete existing databases.
Settings persist in `data/runtime/recording.json` and affect subsequent campaigns.
Both manual experiments and assigned DoE campaigns use this recording pipeline.
Manual experiments, independent DoE campaigns, Dashboard LOOP and BO/IMED recipe
execution use the same general Experiment output folder. The Control Panel field
mirrors that location; its Browse button updates the Settings field. Older saved
Control Panel or SQL recording folders do not redirect current runs. SQL remains
optional for every execution route.

Each campaign execution gets a new folder containing a device/profile/settings
snapshot (`definition.json`), continuous `events.jsonl`, a final `summary.json`, and
one subfolder per run. Run folders contain wide `planned.csv` and `execution.csv`.
Execution columns include UTC, elapsed seconds, each device's commanded target,
measured fields and error state. Samples are saved at polling frequency, independently
of plot refresh. Command attempts and successful batches are separate events;
measured values remain separate from targets. CSV writes are batched approximately
once per second. If Excel locks `execution.csv`, samples continue into a durable
pending journal and the dashboard asks you to close the file. CSV appends retry;
closing the file automatically flushes pending rows without duplicating them.
Keep the app open until saving finishes. Exit is refused while CSV data is pending.
Genuine recording failures (e.g. disk full) still stop execution and are reported.
Optional SQLite commits per event; a power failure can still lose uncommitted queued events.

`recordings.sqlite3` in the output folder is the durable local event database and
upload outbox. Its `events` table stores stable IDs, UTC, kind, run ID, JSON payload
and upload status. The remote SQL table uses the same event IDs to deduplicate retries.
Remote outages do not block hardware execution. Uploads continue after End Session
while the application remains open; restarting the app resumes pending uploads.

For an offline cloud-like demo, set Remote SQL URL to
`sqlite:///C:/absolute/existing/folder/cloud.sqlite3`. The destination must differ
from the local outbox. For PostgreSQL, install optional dependencies:

```powershell
.venv\Scripts\python -m pip install -r requirements-sql.txt
```

Then configure a `postgresql+psycopg://user:password@host/database` URL (including
the provider's required TLS parameters). PostgreSQL needs an existing database
and permission to create/insert into `events`. Its adapter is implemented but has
not been tested against a live server in this workspace. URL credentials are stored
in the local settings file; keep it private. URLs are excluded from run snapshots
and synchronization error messages. Clear Remote SQL URL to use local-only recording.

Use **Choose local demo SQL destination…** to select an existing folder for a
second database (`cloud_demo.sqlite3`). Save, run a simulated campaign, then return
and click **Refresh SQL status** to see local records, pending uploads and demo
destination records. Uploads normally catch up within five seconds; pending should
reach zero. Unit-operation definitions have been deferred and removed from settings.

Execution lifecycle/error messages appear in the existing dashboard log. Rotating
application logs, including diagnostic tracebacks, are retained in
`data/runtime/application.log` (5 MB per file, three backups).

## Setup

The tested development environment is Windows with Python 3.13. Create an isolated
environment, install dependencies, and launch from the repository root:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python MAIN_WINDOW.py
```

For a physical NI-DAQ valve, install `requirements-hardware.txt` and the appropriate
NI-DAQmx native runtime separately. No NI runtime, serial instrument, OPC server,
MATLAB installation, or cloud account is needed for the simulated demo.
The dependency list is a development baseline, not a completed redistribution
license audit or an installer.

## Hardware-free demo

1. Open **Control Panel → Load Simulated Demo**. Existing devices must be
   disconnected, operations finished, and any experiment session ended.
2. Select **Connect All** and wait for completion. Every demo device is marked
   `[SIMULATED]`; none opens a real connection.
3. Leave **Simulation speed** at **1×** and select **Start Experiment**. A prebuilt
   30-second profile runs in real time, followed by the five-second profile hold.
   Choose a higher multiplier explicitly if you want accelerated simulation.
4. Inspect live plots and the Completed / Stopped / Failed status. Application
   dashboard logs receive execution lifecycle events and errors.
5. Use **End Session** before replacing configurations or closing the app.
   End Session is disabled during startup, execution, and cleanup. Stop the run
   and wait for cleanup, or let it complete, before ending the session.

**New Connection → Connection Mode → Simulated** also creates individual simulated
devices. These can coexist with real OPC readers. Mixed real/simulated campaigns
always execute at real time. Save/Load Configuration preserves simulation settings.
Startup and **Reset Defaults** load the bundled simulated crystallization
configuration, including its topology, profiles, and saved experiment. Its
recordings go to `control_panel_runs/crystallization_demo` under the app's data directory.

Simulated types: Masterflex, Ismatec, Huber, JKem, Mettler, NI-DAQ valve, and OPC UA.
Temperature/speed/flow have first-order response and seeded noise. Simulated pump
delivery accumulates in simulated balances. The demo assumes density 1 g/mL and
Masterflex conversion 0.1 mL/min/RPM; these are illustrative, not calibrations.
Simulation settings support `seed`, `noise`, `response_time`, and `fail_after`
(number of accepted target commands before rejection).

## Tests

The running Windows taskbar button uses `assets/CrystaLoop.ico`, with explicit
native small/large icons applied after Windows creates the button. The window
title is **CrystaLoop — Experiment Control Center**. This also applies to Python
and IDE launches. After updating icon code, close existing instances and relaunch.
Rebuild the ICO after replacing the logo with `python -m utilities.build_application_icon`.
`python -m utilities.check_windows_taskbar` opens a five-second, hardware-free
test window and verifies its real Windows icon handles; it requires a desktop.

Tests use Python's built-in unittest runner; pytest is optional:

```powershell
.venv\Scripts\python -B -m unittest discover -s tests -v
$imedTests = Get-ChildItem imed_python/test_*.py | ForEach-Object { "imed_python.$($_.BaseName)" }
.venv\Scripts\python -B -m unittest @imedTests -v
.venv\Scripts\python -B -m unittest imed_python.validation.test_ccas_qmom imed_python.validation.test_numerical -v
.venv\Scripts\python -B -m unittest bo_app.test_workflow -v
```

Optional static checks use `pip install -r requirements-dev.txt` followed by
`python -m ruff check .`. The repository excludes generated notebook checkpoints
from source control. See [the code audit](CODE_AUDIT.md) for cleanup and regression details.

Tests select Qt's offscreen backend and never connect physical devices. Temporary
export fixtures are created inside the repository and cleaned after each test.
Coverage includes profile validation, simulation, command rejection/timeout,
stop-before-start, cleanup failure, ownership, result-file stability, factorial
designs, workbook re-export, and coordinator transitions.

## Architecture

- `gui/pages/`, `MAIN_WINDOW.py`: Qt interface and service wiring.
- `experiment/models.py`: immutable device profiles and experiment definitions;
  campaigns and separate run identities/status. Existing DataFrames are adapted
  at the experiment worker boundary.
- `experiment_dashboard/experiment_dashboard.py`: worker execution and live plots.
  Actuator command failure aborts execution and initiates cleanup. Telemetry is
  best effort. Huber remains connected/regulating after a run, as before.
  The original Matplotlib graph layout, legends and secondary axes are retained.
  A background worker owns an independent Matplotlib Agg figure and renders
  snapshots into images. The GUI only paints the latest cached image. One render
  may be active at a time; later updates are coalesced and stale experiment/theme/
  resize frames are discarded. Lightweight queued slots and a 500 ms display timer
  batch plot updates. Bursts
  coalesce to the newest display sample, with at most 2,000 displayed points per
  signal. Device polling and control are independent of display refresh. This is
  a bounded live view, not a full-resolution recording facility.
- `hardware_devices/simulated.py`: driver-compatible simulation without physical I/O.
- `hardware_devices/hardware_manager.py`: exclusive ownership/reservations.
- `experiment/doe/`, `experiment/profiles/`: sampling, profile generation and OMNI export.
- `experiment/execution_coordinator.py`: separate external OMNI/MODEL file workflow.
- `app/logging_config.py`: worker-safe Qt log delivery.

## Device limits

Open **Control Panel → Device Limits** while no session or connection operation
is active. The table shows each controllable device's minimum, maximum, minimum
nonzero magnitude, and units. Changes apply to validation, manual widgets, and
the stirrer's driver guard; Save Configuration retains them. A minimum nonzero
of zero disables the low-speed gap. Zero remains available as a stop command for
pumps and stirrers. Overrides change application checks, not physical capabilities.
Read-only probes/scales have no target range. Valve targets remain binary.

| Type | Default target range | Minimum nonzero |
|---|---|---|
| Masterflex | −300 to 300 RPM | 0 |
| Ismatec | −1000 to 1000 mL/min | 0 |
| Huber | −50 to 200 °C | 0 |
| JKem | 0 to 800 RPM | 0 |
| Valve | 0 or 1 | 0 |

## Dashboard/sidebar EXP → MODEL workflow

**Settings → Experiment output folder** is the general recording root for Control
Panel Start, independent DoE execution, Dashboard LOOP and BO/IMED Execute recipe.
The Control Panel folder mirrors this setting; its Browse button changes the same
location. Recording / SQL settings displays this shared folder read-only. Older
per-mode recording paths do not override it. Runs receive unique campaign folders.
**Model output folder** holds LOOP modeling stages and executed-recipe dataset
snapshots. Runs without modeling do not create a modeling stage. If general paths
have not been set, the app uses the existing recording default and a data/models
default for modeling. Each start logs the actual recording root.

Define existing Experiment and MODEL folders in Settings and generate a sampled
DoE, then preview and **Apply assignments** in Settings. **Dashboard Start** and
**sidebar Start** launch that assigned campaign in the loop. Its records are saved beneath the
selected Experiment folder, with a unique campaign folder per execution.

The counter reads `run.json` completion markers only within that active campaign.
It validates campaign/experiment identities and counts each successfully executed,
fully saved experiment once. Unrelated folders and `expttsd*.csv` copies do not
change this counter. Stopped/failed runs cannot advance to MODEL. A locked CSV
keeps the transition waiting until the pending data has been saved.
Temporary OneDrive/file-sharing locks on campaign metadata no longer stop loop
monitoring. Unreadable or partially synchronized JSON markers are retried on
later polls; already counted experiments remain counted, and MODEL waits for
readable completion records. A warning identifies the affected file, followed
by a recovery message when it becomes readable. Completed run metadata is
published only when its content changes, reducing unnecessary synchronization.
Locked final run/summary writes also retry before recording is marked complete.

When all expected experiments and the final campaign recording are complete,
the loop creates a unique `_model_<n>_<campaign>` folder in the selected MODEL
root. `campaign_input.json` points to the campaign folder and completed experiment
IDs. With **Settings → Dashboard LOOP modeling → External MODEL file**, the
dashboard waits for an externally produced `MODEL*.csv`, `.mat`, `.json`, `.txt`,
or extensionless completion marker. After it stabilizes, end the session and use
Dashboard/sidebar Start to launch the next batch. Regeneration requires applying
assignments again.

Click the **information icon beside Dashboard LOOP modeling** for the external
model guide. It shows the active MODEL iteration folder, accepted completion-file
names and the loadable next-experiment format. **Save next-experiment template…**
exports the assigned DoE or current Control Panel batch as a Control Panel JSON
configuration. External programs can edit its experiments in device command units
and save next_experiments.json beside campaign_input.json. Publish MODEL.json last
to mark calculations complete; in this version, load the next-experiment file
through Control Panel → Load Configuration and start it explicitly. **Copy AI review
prompt** gives a prompt for reviewing the method, recordings and exported configuration
with an AI. It describes this file handoff and asks for data/configuration checks.

Experiment Definition stage purposes include **Preparing for experiment**, **Dissolution**
and **Seeding**, as well as Charging, Experiment, Equilibration, Wait and cleanup phases.
For manual profiles, use **Control Panel → Run stages…** before generating each experiment.
Save/load configuration and full setup presets preserve the stage labels and intervals.
In **Automatic loop setup → Experiment period**, keep the definition's Experiment and
Equilibration period, select adjacent named run stages, or enter manual start/end seconds
in Recording. The selection applies to both BO and IMED. Sampled durations resolve to
each experiment's own boundaries. Modeling time starts at zero at the selected start;
full recordings and stage-change logs still include the entire run. Follow-up recipes
retain the first starting run's preparation and cleanup around the proposed modeling
period. Supply process masses, concentration and seed information for the start of that
period, after any charging/dissolution/seeding outside it.

### Shared physical inputs, average mixing and inventories

The Control Panel's physical topology is shared with loop assignments. In
**Unit operations / modules**, label source containers as Solution, Solvent,
Antisolvent, Seed suspension or Cleaning solvent, and identify the model reactor.
Device cards also provide an **Experiment role** and **Measured quantity**.
Automatic loop assignment follows these MSI connections and positive pump
directions, reuses matching applied DoE assignments and resolves unique probes.
Choose an explicit device/channel when more than one candidate remains. The
**Used values** tab shows the assignments actually selected and their origin.

**Starting experiment overrides** change a named experiment's model inputs. They
do not operate pumps. Defined preparation charges can instead supply starting
liquid masses automatically; explicit named overrides take precedence. Pump
calibrations convert commands to grams. Seed-dose generation can reuse the
calibration already loaded on the seed pump, using only the selected operating
RPM. Manual g/s calibration remains available. Unseeded IMED inputs accept
**0 g seed and initial D10/D50/D90 = 0/0/0**, with no initial seed population.

In **Automatic loop setup → IMED design**, optional average mixing uses
`epsilon = Np * (average_RPM / 60)^3 * diameter_m^5 / average_volume_m3`.
RPM and liquid volume are averaged over the selected modeling period, then one
constant is placed in each experiment's existing `PowerRatio` input. No stirring
or volume profile is introduced into the PBM equations. When unchecked, this
loop input is 1. Power number and impeller diameter are campaign constants.
Fixed volume is entered in mL. Charged-volume mode uses starting liquid amounts
plus additions, preferring AS scale data; calibrated pump delivery supplies the
fallback. Component densities give an additive-volume estimate. The power-number
relation and specific-power convention are described in the
[impeller power reference](https://proquipinc.com/wp-content/uploads/2020/07/Impeller-Design-and-Performance-Considerations-for-Industrial-Tank-Agitators.pdf)
and [stirred-tank mixing study](https://www.sciencedirect.com/science/article/abs/pii/S0263876213002682).

The existing PBMs apply this power input to **secondary nucleation and
agglomeration**. The initial campaign records which stirring/volume inputs varied.
For varying power, existing non-degenerate candidate power-exponent bounds can be
enabled for estimation; disabled mechanisms and fixed scientific bounds are
retained. Use consistent mixing inputs across fitted data and independently vary
RPM and volume to distinguish their effects. SA exposes initial-volume variation
when present in the initial campaign and recomputes scalar mixing for its draws.
BO exposes scalar initial charge volume and editable **Stirring** stages. Volume
recommendations require executable preparation charging profiles; AS-dependent
volume changes are controlled through AS rate/time. Importing a recorded campaign
defaults to each run's own experiment period. Saved command profiles retain
stirring through final holds and remain available when a case is moved.

Use **Control Panel → Reservoir amounts…**, also accessible in Settings, to enter
available material in grams. Blank amounts are untracked. Enter source density
for volumetric pumps without a mass calibration. The dashboard shows remaining
defined reservoirs and updates during additions. Stock is retained across runs and
reloading the same refill; change the amount after refilling to start a new stock
record. Each next experiment is checked before its commands are submitted. An
insufficient reservoir blocks that experiment and identifies available/required
grams. Source scales correct calculated draw where configured; otherwise delivery
is calculated from the pump calibration and commands.

The experiment window and `execution.csv` show **system_mass_g**: starting reactor
contents plus deliveries. Opening a drain valve is treated as complete discharge;
later additions accumulate again. Enter **Starting reactor mass** for material
already present before a manual run. Undefined connections/calibrations leave an
unavailable mass indication rather than inventing delivered grams.

The DoE assignment preview explicitly lists read-only monitors included for
recording/live plots and any disconnected probes it excludes. A saved empty
monitor list restores the checkbox as off; enabling it selects all connected
read-only monitors. This prevents an invisible empty filter from excluding FBRM
and D10/D50/D90 channels while the checkbox appears enabled. Those channels are
measurements and do not appear in pre-run actuator-profile plots.

### Experiment Definition timeline

Open **Experiment Profile** in Settings and choose **Device timeline**. Add device
lines with custom names and compatible assignments, then add named stages. Time
runs downward and device lines run across the page. Drag Hold, Ramp, Mass addition,
Equilibration, or valve actions onto cells. Double-click an action to edit its
values, or a column header to edit its device line. Drag actions to move them and
drag stages in the list to change their order. Several actions in one row run in
parallel. Reuse the same pump line during charging, antisolvent addition, cleaning,
and rinsing. An empty pump or valve cell is off. An empty temperature or stirring
cell holds the preceding target.

**Timeline time units** defaults to minutes and also offers seconds and hours.
Stage/device-profile durations, the displayed experiment horizon and heating/cooling
rate limits use matching units (°C/min, °C/s or °C/h). Switching units converts the
display and preserves existing timing and limits, including saved presets. Drafts
can be saved with rates above the configured limits; their warnings appear in the
application log. Generating the DoE checks those limits and reports the requested
temperature change, actual rate and minimum required duration.

Device Start/End fields have **Equal to previous** checkboxes for temperature,
pumps, stirrers and valves. Each linked
value resolves from the preceding section's final target separately for each
generated experiment (the first section uses the line's Initial target). For
example, alternate the cooling End between 15 and 20 °C and check Equal to previous
on the equilibration Hold: it stays at 15 or 20 °C in the matching experiment.
Linked values do not add independent DoE dimensions. New ramps link
their Start by default; the Equilibration palette action links its target.
For a linked Mass addition, the preceding pump flow is maintained and its mass
is calculated from the new section's duration. Valve links retain a binary state.

The editor has a wrapping toolbar, a scrolling stage sidebar and a larger
timeline. Each distinct section boundary has enough vertical space to show its
actions, including short independent sections. The window can be resized or
maximized; long definitions scroll instead of compressing the stage controls.

Click **Check one representative experiment** at any time to plot the current
definition before saving or generating the full DoE. It uses Value for Fixed/DoE
controls and Minimum for Alternate controls (experiment 1), resolves linked
targets, and uses the selected time unit. Rate warnings remain visible while the
draft is plotted. Changing the definition and clicking again refreshes the plot;
the saved definition and previously generated experiments remain unchanged.

An action's **Action lasts until** can span several shared stages, or continue to
**End of experiment (longest profile)**. Define a constant stirrer once in the
first stage and use that end option. Shared stages may contain any subset of
device actions; overlapping actions on one device are rejected.

Use **Edit device profile…**, also available from a device column header, for an
independent sequence of Holds, Ramps or Mass additions. Give each device its own
section count, durations and Fixed/DoE/Alternate controls. Saving an independent
profile replaces that line's shared-stage actions. Generation first resolves
every profile separately, takes the longest total duration, and synchronizes all
devices onto the combined time points. Shorter profiles retain their final target
until the common end. A Mass addition completes at its own section's end and
extends at zero flow to preserve the requested mass. Add an explicit zero-flow or
closed-valve section when you want those states at the end of a non-dose profile.

Set each duration and target to **Fixed**, **DoE** with bounds, or **Alternate**
between its bounds. Mass additions specify grams and derive the calibrated flow
from the stage duration. Temperature actions select **Jacket** or **Internal vessel
(Process)** control, including mode changes between stages. **System settings**
retains the existing ramp and addition limits. **Import current component profile**
creates an editable timeline from the existing component design. Set the imported
stage purposes before using it for modeling. **Component design** remains available
in the definition source selector for existing setups.

Save, generate the DoE with the existing sampling choices, then apply device
assignments. Presets retain the recipe and exact generated profiles. CSV export,
the native runner and recorded campaign viewer use the same compiled profiles.
Keep Experiment and Equilibration stages together to define the modeling window.
Native modeling follow-ups retain the timeline's charging and cleanup profiles.
Independent profiles without shared stages use the full experiment as their
modeling window. With shared stages, their purpose fields define that window.

**View DoE Experiment Profiles** uses the original combined plot style for both
component and timeline designs: matching colors, line styles, colored axes and
legend, with actual units for each signal. In the **Control Panel / Control
Center**, **View Experiment Profiles** opens the same viewer for every manually
generated experiment. Previewing does not connect devices or execute experiments.

### Automatic BO and IMED

Choose **Bayesian optimization** or **IMED** under **Dashboard LOOP modeling**.
Generate or load a DoE and apply its device assignments, or prepare a Control
Panel batch and map its device IDs to DoE roles on the **Devices** tab. Connect
the devices and monitors, then choose writable Experiment and MODEL folders. Open
**Automatic loop setup…** to set the number of **Follow-up experiments** and edit
the **Recording**, **Process inputs**, **Devices** and **Support profiles** tabs.
**Used values** shows the applied assignments, starting profiles, resolved process
inputs, recording mappings and modeling settings for the selected starting batch.
The **Source** column distinguishes inherited values from overrides. Return to
**Used values** to see edits before saving.
The **IMED design** tab is available when IMED is selected. **Save setup preset…**
retains these choices for the next session. BO
targets and stage templates come from the BO page. IMED uses the case, solver,
PE settings and response weights loaded in IMED, plus the size-sample choice in
Settings. Use the [simulated configurations](examples/configurations/README.md)
to try the workflow without instruments.

**Dashboard Start** first checks the setup. Once the initial batch is saved, BO
adds the new observations to the selected prior runs, fits its surrogates, and
proposes a recipe within the configured bounds. IMED refits the candidate models
on the accumulated recordings. IMED applies the original MATLAB F-test and
parameter t-tests, then retains models within `imed.delta_aic` of the lowest total
AIC among candidates that passed those checks. Multiple survivors trigger model discrimination. One survivor
with usable covariance triggers D-optimal design. If no candidate passes, the
loop saves the fit diagnostics and reports why it stopped in the dashboard log.

The loop opens the selected modeling page and fills it from the saved recordings.
IMED displays Experimental Input and automatically opens Parameter Estimation,
followed by the applicable design window. BO shows the input table, then the
predictions and proposed recipe. The dashboard log follows the source selection,
data import, calculations, files written and follow-up launch.

Each MODEL folder contains `inputs/` with the completed run CSVs, a `model.log`,
and BO or IMED input CSVs. A round that produces a recipe saves
`recommendation.json` with the validated device profiles, units, process inputs
and campaign ID. IMED saves `fit.imed.json` after parameter estimation and exports
the fitted parameters and covariances as CSVs. After the last follow-up, IMED
refits the retained candidates on the full dataset and saves a final `fit.imed.json`.
**IMED Loop stages** lists the evaluated and retained candidates in each named stage.
Open a stage's PE results there or load its `pe_01.imed.json`, `md_01.imed.json` or
`do_01.imed.json` from the corresponding MODEL folder. The IMED Case status and input
CSVs record the stage that produced each automatic follow-up. CrystaLoop launches each
follow-up after recording and hardware cleanup, rechecks
device connectivity and limits, then repeats until `max_followups` is reached.
**Stop** discards any pending launch and requests cancellation from the model
worker. An active solver call may take time to finish. End the final session to
leave the hardware available for another campaign.

**BO Loop stages** shows the training runs, their source stages and the predictions
behind each proposal. Load a saved `bo_01.json` (or a later stage) with **Load BO setup**
to revisit its data and settings. BO also saves a final observations stage after the
last follow-up.

For one-off execution, use **Execute recipe** beside the BO proposal or an IMED
design result. CrystaLoop reuses the applied DoE assignments or the Control Panel
assignments in **Automatic loop setup → Devices**. It checks connections and
limits, then runs one experiment. Once the recording is complete, CrystaLoop adds
the measurements to the BO training set or IMED Experimental Input and saves an
updated dataset in the MODEL folder. If no MODEL folder is configured, the snapshot
is saved in `model_inputs` under the campaign recording folder.
Previous IMED designs remain available in their design window and are labeled
when they were generated from earlier inputs.

For physical runs, use **Recording** to select measured CSV columns and set the
crystallization window, concentration multiplier and scale direction. Use
**Process inputs** for starting masses, densities, shape factor and seed mass.
Blank fields can inherit values from the first IMED process record. The override
table sets individual values by starting experiment name. Set **Follow-up
crystallization start** to place the recommended profiles after charging.
Use **Support profiles** for charging, stirring, discharge, or cleaning commands.
Enter time and target points, starting at elapsed zero, in the displayed role
units. Keep pump and valve support profiles at zero at their end. For variable
seed mass, enable calibrated seed-pump delivery in **Devices** and enter the
seed D10, D50 and D90 in **Process inputs**.
Check that the configured startup conditions and auxiliary profiles deliver the
same physical conditions the optimizer assumes.
If both Control Panel and assigned DoE profiles are present, Dashboard Start
asks which batch to use. For a physical Control Panel batch, turn off simulated
PBM telemetry in **Devices**. For a physical DoE, apply the actuator and monitor
assignments after connecting the instruments.

Dashboard/sidebar **Start** offers **Use Control Panel experiments** when the
Control Panel has a manual batch. This runs the batch using the configured LOOP
folders and advances to MODEL after recording completes. The batch size is
inferred when the Settings experiment count is blank. The Control Panel's own
**Start Experiment** button and the standalone Settings DoE button run independent
sessions. Use **Skip to next experiment** in the
running dashboard to save the current run as skipped and idle the actuators before
starting the next run. The loop lists skipped runs separately from MODEL inputs.
Use Stop to cancel an active loop, then wait for hardware cleanup before Reset or
End Session.

## Scope and next hardware gate

Release A supplies the shared model, simulation, initial logging and regression
coverage. Release B supplies DoE-to-device assignment, preview, execution and CSV exports.
Release C supplies continuous CSV/local SQL recording, optional SQL synchronization,
and rotating application logs. Live PostgreSQL deployment
and physical-device acceptance still require their respective environments.

Before accepting physical execution, use the instrument PC to check each driver's
connection, setpoint, feedback, calibration, timing, stop, disconnect and cleanup.
The executor waits for in-flight hardware calls before sending cleanup commands;
an unresponsive driver can delay shutdown. The dashboard refuses to close if its
worker has not terminated. Its stop button is a software stop request, not a
replacement for instrument interlocks or a physical emergency stop.
