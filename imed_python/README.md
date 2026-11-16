# IMED standalone PySide6 port

From the workspace root, run `python -m imed_python`. Use **Import MATLAB (.mat)** to open a saved IMED case with its experiments, process inputs, solubility, fitted models, and mesh settings. You can also open a file at startup with `python -m imed_python "C:\path\to\case.mat"`. Use the MATLAB `-v7` save format for import.

The landing screen follows MATLAB's `ControlPanel.m` and uses its workflow diagram. **Initial DoE / Experimental input** inspects saved data. **Model Configurations** lets you edit parameter values, bounds, and estimation flags. Saving in that dialog applies changes to the current session and clears fitted results for changed models. Use **Save Model** on the control panel to write a new MAT file. **Digital Design** opens the forward simulator for the loaded case. Select an experiment and model and click **Simulate** to see count, concentration, the final volume-based size distribution, and D10/D50/D90. The percentile tab compares model predictions with measured values when the case contains them. For direct access, run `python -m imed_python --simulate`.

The control panel's **PBM solver** selector applies to Digital Design, Parameter Estimation, Model Discrimination, and D-Optimal Design. **JAX PBM** uses JAX's default device. **JAX PBM (CPU)** and **JAX PBM (GPU)** select a device explicitly. The GPU option reports a setup error if JAX cannot initialize a GPU. Digital Design also lets you choose a solver for a single simulation and reports the device used. JAX compiles on the first run for a given input shape and uses float64. **SciPy PBM** is the default for cases without a saved solver choice.

For GPU use on Windows, set up [WSL2 with Linux GUI support](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) and follow the [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html) for your GPU and driver. In a WSL2 Python environment, install NumPy, SciPy, PySide6 and Diffrax alongside GPU-enabled JAX. Run `python -c "import jax; print(jax.devices('gpu'))"` to verify that JAX sees the GPU, then launch `python -m imed_python` from the workspace root and select **JAX PBM (GPU)**. The packaged Windows app uses the native SciPy solvers.

With a SciPy solver selected, **CPU workers** controls how many candidate simulations can run at once during the global search in Parameter Estimation, Model Discrimination, and D-Optimal Design. The default of one uses the serial search. With multiple workers, differential evolution updates its best candidate once per generation, so it may take a different search path. Local least-squares refinement and individual Digital Design simulations each run as a single job. The headless `fit_model` and `optimize_recipe` functions accept a `workers` argument for the same parallel search.

The simulator uses fitted parameters where present and fixed values from the parameter table. The headless entry point, `imed_python.solver.simulate`, accepts a 35-element IMED parameter vector, experiment and process dictionaries, solubility coefficients, and optional grid settings. Set `settings["Numerics"] = "jax"` and optionally `settings["JaxDevice"]` to `"cpu"` or `"gpu"`. JAX mode uses a step limit, configurable with `settings["MaxSteps"]`. The JAX solver was tested with JAX 0.10.1 and Diffrax 0.7.2 on CPU.

**Parameter Estimation** runs one or all models against the loaded experiments. Set count, concentration, and size weights and choose full CSD or D10/D50/D90 for the size term. The optimizer uses SciPy differential evolution followed by bounded least squares. Apply the fitted parameters and use **Save Model** to persist the parameters and fit diagnostics.

**Model Discrimination** requires at least two fitted models. It searches for a recipe that maximizes weighted pairwise separation between predicted responses. **D-Optimal Design** requires a fitted model with finite parameter covariance and at least two selected responses. It searches for a recipe that maximizes the log determinant of the sensitivity-based Fisher information matrix. Both can export the recommended profiles and seed mass as CSV.

The design search supports one temperature ramp and one antisolvent pulse. Thermocycling, MATLAB design-history persistence, experimental input editing, and Digital Design sensitivity analysis are pending. PE reports per-model fit diagnostics but does not yet reproduce the MATLAB cross-model F and t test decisions. Numerical parity with MATLAB is still unverified.

Run `python -m unittest imed_python.test_solver imed_python.test_estimation imed_python.test_design` for the local calculation checks.

The [PBM numerical validation](validation/README.md) includes the independent QMOM implementation, a nine-case grid-refinement study, and the saved comparison trajectories.

For archived MATLAB fits, **Archived MATLAB PE** uses the historical parameter-estimation discretization and an ode45-style step controller. The mode retains the historical mass-balance defects. **SciPy PBM** and **JAX PBM** use the corrected conservative discretization. Parameters calibrated with one scheme may need refitting for use with the other.

The archive comparison runner recovers MATLAB predictions from saved weighted residuals and writes one summary row per model, experiment, and solver mode. It reports count shape after peak normalization, concentration at the recorded times, and final volume-based D10/D50/D90.

```powershell
python -m imed_python.validation.compare_matlab "C:\path\to\case.mat" --output comparison
```

## CrystalLoop campaign recordings

Use **Import CrystalLoop** and select the campaign directory containing `definition.json` and the individual run folders. IMED reads `execution.csv` for the completed runs whose recorder has finished writing. Select the runs, confirm the recorded channel mappings, and enter the process inputs shared by those runs. Use the elapsed-time window to exclude charging, pump return, or cleaning. Import groups with different initial conditions separately.

The importer reads temperature, concentration, FBRM counts, D10/D50/D90 and the antisolvent scale directly from the CSVs. Set the concentration multiplier to convert recorded readings to g/g solution. For example, use 0.01 for a mass percentage. Supply an initial concentration explicitly if the instrument lacks a reading at the start. IMED retains the size percentile histories and takes the last simultaneous, ordered triplet as the final size measurement. It leaves the full measured size distribution empty.

For antisolvent, select the direction appropriate to the scale. A reservoir scale normally decreases. IMED median-filters the scale, estimates nondecreasing added mass, and derives a nonnegative feed profile in g/s with the same integrated mass. **Experimental Input → Recorded channels** shows the imported readings, the added mass estimate, and the derived feed.

For modeling, open a model case with the appropriate kinetics and solubility first, then import the campaign. The import replaces the experimental records for the session and retains the model configuration. CSV campaign import requires no MAT conversion. **Save Model** is disabled while a campaign is active because native campaign-project saving is pending.
