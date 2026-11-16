# Standalone Bayesian experiment selector

Run from the workspace root with `python -m bo_app`. Dependencies are PySide6, NumPy, pandas, PyTorch, and scikit-learn. Training and ranking run locally on the CPU.

1. Open `generated_profiles.csv` from the CrystalLoop DoE export. Required columns are `experiment`, `time_s`, `T` (°C), and `AS` (g/h). Extra recipe columns are preserved when exporting the selected recipe.
2. Prepare an outcomes CSV with one row per generated experiment. Use the same experiment names in both files.

   ```csv
   experiment,status,seed_amount,size,c_initial,c_final,c_equilibrium,batch_time_s
   Exp1,completed,0.50,85,0.12,0.04,,3200
   Exp2,completed,0.75,78,0.12,,0.05,3000
   Exp3,candidate,1.00,,,,,
   ```

   Use consistent seed units and one size statistic such as D50 across the campaign. `pending` and `excluded` recipes are skipped during selection. Missing batch time uses the planned profile duration. For yield, express concentrations as dissolved solute mass per unit of initial carrier mass. Convert readings to that common basis using carrier masses when antisolvent addition changes the mass. The app computes recovery as `(c_initial - c_final) / c_initial` and substitutes the supplied equilibrium concentration for a theoretical estimate when a final measurement is missing.

3. Select minimize, maximize, or target for size. Set any objective weight to zero to omit it. Zero exploration ranks by predicted loss. Increasing the exploration weight favors candidates with greater posterior uncertainty.
4. Rank the candidates and export the top recipe.

Two single-profile VAEs use the architecture from the supplied temperature and antisolvent notebooks and train on the campaign trajectories. The Gaussian process uses both latent vectors, seed amount, and planned duration. Duration is a descriptor derived from each recipe since VAE inputs use normalized time. Candidate search is restricted to generated recipes. At least one completed experiment with values for all enabled objectives is required, and rankings with only one observed result are driven by exploration.

The standalone workflow ends at recipe export. A future integration can pass `Window(start_experiment=callback)` to route a reviewed recipe through its validated experiment runner.

Run `python -m unittest bo_app.test_workflow` for the local smoke test.
