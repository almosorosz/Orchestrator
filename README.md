# CrystaLoop 0.4.0 — Campaign archive, recovery, presets and Windows release

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
Select a campaign to see every experiment as queued, running, saving, completed,
interrupted, or not executed. Completed experiments can be inspected while the
remaining experiments run or wait in the queue.

Ctrl/Shift-select up to eight executed experiments and **Plot selected**
to compare target and measured values against each experiment's elapsed seconds.
Selected experiments overlay on shared device panels, with a distinct color for
each experiment and line styles distinguishing target/measurement fields. Compact
controls and short selection tables leave most of the page for the graphs.
Choose one device or all devices. **Fit width** and zoom control the cached plot.
Enter elapsed seconds in **From (s)** and **To (s)** before plotting to limit the
time window. Leave either bound blank to keep that end open. **Clear plots**
clears the display and discards pending plot results.
Each click reads a snapshot; plotting a running experiment does not follow it live.
Read/plot/export work runs off the GUI thread, separately from acquisition. Display
traces retain up to 4,000 points using extrema-preserving reduction; CSV recordings
and exported files retain every saved sample. Torque and valve commands use a
separate axis where their units differ. **Export CSVs** copies fully saved,
completed experiments to a new directory. **Open folder** opens their files.

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
SQL connection URLs are included; treat presets
containing credentials as private configuration. Loading requires an idle session,
restores disconnected devices. A saved generated DoE restores without sampling again,
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

**Profile encoding** uses the trajectory VAE architecture from
`BO_test_V2_single_profile`. It fits the selected traces and uses the encoder's
posterior means. You can load `trajectory_vae_T.pt` and `trajectory_vae_AS.pt`
exported by the notebook to use those weights. Compare the source trace, eight
input knots and decoded profile before reviewing a proposal. This tab reports
reconstruction error and integrated antisolvent dose. Seed and constant
setpoints use scalar values. Missing channels must be consistent across the
selected runs.

**Propose experiment** shows the decoded recipe, input history, and one-dimensional
GP projections through the proposal. **Observed inputs** shows standardized
values together or any input in its own units. **GP projections** shows the
predicted mean, a predictive band at ±1.96 standard deviations, the acquisition
for the weighted objective, and the actual measured points when available. The
output filter uses the selected margin and requires the full predicted interval
to meet every enabled bound. With small datasets, use the plots and reconstruction
errors to judge whether the proposed recipe is suitable for experimental review.

**Save BO setup** includes the selected runs, settings and any imported VAE
weights. **Save proposed recipe** exports decoded profiles and predictions as
JSON for review. The Windows build includes BO when PyTorch and scikit-learn are
both installed in the build environment.

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
automatic for every experiment under `data/runs`; choose a different output folder if needed.
SQL is independently optional: uncheck **Also save to SQL** for CSV/JSON-only
recording. New installations default to SQL off. Enabling SQL creates the local
database; leaving Remote SQL URL blank keeps it local-only. Turning SQL off stops
future uploads but does not delete existing databases.
Settings persist in `data/runtime/recording.json` and affect subsequent campaigns.
Both manual experiments and assigned DoE campaigns use this recording pipeline.
Manual experiments use the separate **Control Panel experiment folder** (default
`data/control_panel_runs`), saved with the Control Panel configuration JSON.
Older configurations without a folder use this separate default. Changing the
Settings recording folder does not change the Control Panel folder. Independent
Settings DoE runs use the recording folder in Settings; loop campaigns use the
Experiment path in LOOP. SQL remains optional for all three execution routes.

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
```

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
IDs. The dashboard enters MODEL and waits for an externally produced `MODEL*.csv`,
`.mat`, `.json`, `.txt`, or extensionless result to stabilize. The app does not
launch MATLAB/IMED/BO yet. `MODEL.csv` is currently a nonempty completion marker;
its contents are not parsed as recommended profiles. After the file stabilizes,
MODEL becomes inactive and OMNI monitoring resumes, ready for the next campaign.
The previous campaign is not replayed automatically. End its session, then use
Dashboard/sidebar Start to launch the next assigned DoE. Regeneration requires
applying assignments again. Defining and
consuming an optimizer recommendation format remains part of Release D.

Dashboard/sidebar **Start** offers **Use Control Panel experiments** when the
Control Panel has a manual batch. This runs the batch using the configured LOOP
folders and advances to MODEL after recording completes. The batch size is
inferred when the Settings experiment count is blank. The Control Panel's own
**Start Experiment** button and the standalone Settings DoE button run independent
sessions. Use **Skip to next experiment** in the
running dashboard to save the current run as skipped and idle the actuators before
starting the next run. The loop lists skipped runs separately from MODEL inputs.
Use Stop to cancel an active loop, then wait for hardware
cleanup before Reset or End Session.

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
