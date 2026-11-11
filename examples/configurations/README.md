# Simulated crystallization system

## Real-hardware variant

`crystallization_system_real.json` contains the same six unit operations and 15
devices with every device set to Real hardware. Load it through Control Panel →
Load Configuration. Simulation parameters and simulated-name suffixes are removed.
Its separate recording folder ends in `crystallization_real`.

Serial ports and OPC endpoint/node IDs remain blank; enter actual values before
connecting. Masterflex node addresses and NI-DAQ channels are inherited placeholders
and must match the installed hardware. Supply actual pump calibrations as needed.
IR and EasyViewer entries are read-only OPC channels; this file adds no conversion
model, so they require nodes already providing the intended quantities. Loading the
file does not connect devices or start experiments. Hardware communication has not
been tested with this export.

## Simulated variant

Load `crystallization_system_simulated.json` using **Control Panel → Load
Configuration** (not Settings → Load setup preset). End any active session first.
Loading restores disconnected simulated devices; it does not start an experiment.
The Control Panel recording folder is saved in the file and can be changed after
loading. It is separate from the Settings/LOOP folder.

Open **Unit operations / modules** to inspect the six unit operations in module
`m1`. Device-card connectivity is already assigned. **Generate MSI / connectivity
graph** produces the current application representation. A matching MSI text file
is included for inspection.

| Device ID | Hardware type | Connection in the physical definition |
|---|---|---|
| PAS1 | Ismatec | AS1 → C1 |
| PSOLV1 | Masterflex | SOLV1 → C1 |
| PCLEN1 | Masterflex | CLEN1 → C1 |
| PSOLU1 | Masterflex | SOLU1 → C1 |
| PSEED1 | Masterflex | SEED1 → C1 |
| WAS1 | Mettler, read-only | AS1 |
| SSOLU1 | JKem stirrer | SOLU1 |
| SSEED1 | JKem stirrer | SEED1 |
| FBRM1 | Simulated OPC, read-only | C1; counts placeholder |
| IR1 | Simulated OPC, read-only | C1; concentration placeholder |
| D101 | Simulated OPC, read-only | C1; EasyViewer D10 placeholder |
| D501 | Simulated OPC, read-only | C1; EasyViewer D50 placeholder |
| D901 | Simulated OPC, read-only | C1; EasyViewer D90 placeholder |
| T1 | Huber, controlled | C1; Process control selected |
| V1 | NI-DAQ valve, controlled | C1; drain valve |

Unit operation codes are `C1` (crystallizer), `AS1` (antisolvent), `SOLV1`
(solvent), `CLEN1` (cleaning solvent), `SOLU1` (solution), and `SEED1` (seed).
Pump source → target means positive/clockwise direction. No crystallizer stirrer
was added because only solution/seed container stirrers were requested.

All hardware connections are simulated; no OPC server or serial/NI driver is
opened. Each OPC entry is one scalar channel, so EasyViewer is represented by
three entries. The current simulator supplies generic signals, not instrument-
specific counts, concentration, calibrated size distributions, or ordered D10/
D50/D90 values. No IR/EasyViewer model or conversion is applied. The demo scale
uses the existing generic pump coupling, not an AS-container depletion model.
Connectivity currently describes the graph; it does not drive a physical process
simulation or gravimetric flow controller. The valve attaches to C1; a drain
destination/valve flow edge is not supported by the current connectivity UI.

Profiles and experiment batches are intentionally empty: define your intended
targets and timing before executing. Pump/stirrer defaults are zero, the valve
default is zero, and Huber's default is 25 °C. No pump calibration was invented.

The file uses the current MSI mapping: `U_<unit code>`, `R_<read-only device ID>`,
`C_<controlled device ID>`, and `U_P_<pump ID>` / `C_F_<pump ID>` for pumps.
Stirrers generate a separate `U_S_<device ID>` unit with `C_SR_<device ID>` stir-rate
control and `R_TQ_<device ID>` read-only torque. Pumps and stirrers share the same
square representation as the explicitly defined unit operations. Grey dotted
stirrer-to-container connections describe placement, not material flow. The
separate MSI-ID field and notebook-style graph improvements are not implemented
by this configuration.

To use real devices later, replace connection fields and switch the required
devices to Real hardware, then supply actual calibrations/OPC nodes. The empty
OPC/serial fields in this file are placeholders for simulated mode only.
# IMED validation DoE

Load `imed_validation_doe.json` in **Settings → Load setup preset**. In **Control
Panel**, connect the simulated devices. Return to **Settings → Assign devices /
preview DoE** and review the actuator assignments and the rows prefixed **IMED:**.
Select **Preview / validate**, **Apply assignments**, and **Start DoE independent
from loop**.

The three runs are seeded cooling, unseeded cooling with antisolvent, and seeded
isothermal antisolvent addition. The PBM uses the solubility coefficients in
`imed_python/validation/cases.py`. The simulated count channel is the PBM
particle count per liquid volume. The size channels are volume percentiles and
are empty until particles are present.

The crystallization clock starts at the initial temperature. The two seeded runs
command the seed pump at that instant and the PBM treats the seed dose as an
initial condition. Final temperature is reached by 35 s and held through 60 s.
The valve then discharges the batch, the cleaning pump adds rinse solvent, and
the valve drains the rinse. Those operations stay in the recorded timeline.

After recording finishes, choose the campaign folder in **IMED → Import
CrystalLoop**. IMED reads the stored device assignments and uses only the 0–60 s
crystallization interval. It loads the seed distribution, process inputs,
solubility and starting kinetic model together, so you can open **Parameter
Estimation** without a separate MAT file. The default recording root is
`data/imed_simulated_runs`.
