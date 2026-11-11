# IMED PBM numerical validation

The corrected high-resolution finite-volume solver passed all nine benchmark cases at 80, 160, and 320 cells. Each run used the same 60-second experiment and the same 100 output times as its independent three-node quadrature method of moments (QMOM) reference.

## Benchmark matrix

| Operating profile | Growth only | Growth + nucleation | Growth + nucleation + agglomeration + breakage |
| --- | --- | --- | --- |
| Cooling, no antisolvent | Pass | Pass | Pass |
| Antisolvent, fixed temperature | Pass | Pass | Pass |
| Cooling and antisolvent | Pass | Pass | Pass |

The growth and nucleation cases use monotonic cooling when temperature is varied. The full cooling and combined cases include a late reheat. The antisolvent-only full case uses a late high-feed dilution stage at constant temperature. These stages create undersaturation because IMED activates the modeled breakage process in that regime. The runner checks that both agglomeration and breakage occur in the full cases in both implementations.

The benchmarks use size-independent growth, primary nucleation at zero size, a constant agglomeration kernel, and Gaussian-in-volume daughter density. Dissolution is disabled to isolate breakage. This suite covers those kinetics and operating profiles. Other kernels, secondary nucleation, dissolution kinetics, and comparison to MATLAB or experimental data remain outside this validation.

## Results

Errors below are percentages. The time-trajectory error is the largest absolute difference divided by the largest absolute reference value for that trajectory. The moment column is the worst such error among M0 through M5. Mass drift is the largest absolute change in dissolved solute plus solid mass, divided by the initial solid mass.

| Operating profile | Mechanisms | Worst moment error, 320 cells | Count error | Concentration error |
| --- | --- | ---: | ---: | ---: |
| Cooling | Growth | 0.04607% | 0.04104% | 0.002285% |
| Cooling | Growth + nucleation | 0.02584% | 0.01775% | 0.000796% |
| Cooling | All four | 0.02217% | 0.02187% | 0.000265% |
| Antisolvent | Growth | 0.04399% | 0.04104% | 0.000259% |
| Antisolvent | Growth + nucleation | 0.02689% | 0.03297% | 0.000095% |
| Antisolvent | All four | 0.02931% | 0.03936% | 0.000231% |
| Combined | Growth | 0.04566% | 0.04104% | 0.001016% |
| Combined | Growth + nucleation | 0.02540% | 0.02320% | 0.000298% |
| Combined | All four | 0.02313% | 0.02836% | 0.000278% |

| FVM cells | Cases passed | Worst moment error | Worst mass drift |
| ---: | ---: | ---: | ---: |
| 80 | 9 / 9 | 1.7004% | 0.0000660% |
| 160 | 9 / 9 | 0.2892% | 0.0001432% |
| 320 | 9 / 9 | 0.0461% | 0.0001637% |

Before correction, five of the nine 160-cell runs failed. Across the three full-mechanism runs at that resolution, worst M3 error decreased from **2.77% to 0.0571%** and worst mass drift from **8.29% to 0.000143%**. The largest change when tightening QMOM integration tolerances by 100× was **0.0000420%**. This refinement measures time-integration error within the chosen closure.

## Corrections

- Coupled solute transfer to the same finite-volume solid-volume flux used by the population balance, including the nucleation boundary. Solid leaving the upper grid boundary remains a loss from the modeled size range.
- Replaced the agglomeration birth and death accounting with adjacent volume pivots. One event consumes two parents and creates one aggregate, and its volume is conserved. Equal-bin pairs receive the half factor for the whole event.
- Prevented negative reconstructed density at the outflow face. The solver also honors supplied integration tolerances.
- Added a check for signed feed profiles that empty the liquid inventory. Some bundled experiments contain return segments that trigger this diagnostic. Their feed records were left as supplied.

## Imported cases and configurable QMOM

The configurable QMOM comparison uses the same finite-volume seed conversion as SciPy and JAX. It places each seed bin's count at a volume-equivalent size to preserve both the initial particle count and the seed mass. This avoids integrating a measured volume distribution divided by length cubed all the way to zero, which can diverge for data with a positive near-zero slope. The independent analytical reference used for the nine synthetic benchmarks still uses its prescribed continuum seed.

QMOM computes collision and growth terms using up to three quadrature nodes reconstructed from six moments. Its node-based percentiles should not be used as a reference for percentiles from the full PBM distribution. When an integration trial leaves the size domain or fails moment inversion, the solver retries with a smaller maximum time step. If it still cannot finish within its trial and time budgets, the comparison reports the reference as incomplete. Widening the mesh can check sensitivity to the domain but does not resolve all quadrature failures.

## Reproduce

From the workspace root, run:

```powershell
python -m imed_python.validation.run --output imed_python/validation/results
python -m unittest imed_python.test_solver imed_python.test_estimation imed_python.test_design imed_python.validation.test_numerical
```

[QMOM reference](qmom.py) uses a Hankel generalized eigenproblem to obtain three Gaussian quadrature nodes and weights from six moments. It integrates moment and liquid balances with DOP853. [Case definitions](cases.py) supply the common initial conditions and schedules. [Runner](run.py) records full trajectories and tests each moment, count, concentration, mass drift, and mechanism activation against its thresholds.

- [Summary by operating profile, mechanism, and resolution](results/summary.csv)
- [All FVM and QMOM trajectories](results/trajectories.csv)
- [QMOM integration refinement](results/reference_refinement.csv)
