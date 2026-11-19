"""Freeze initial-campaign operating inputs for later BO and IMED sensitivity work."""
import numpy as np


def initial_input_groups(processes):
    def varies(key):
        values = np.asarray([p.get(key, np.nan) for p in processes], dtype=float)
        return bool(len(values) > 1 and np.isfinite(values).all()
                    and np.ptp(values) > 1e-6 * max(1., float(np.max(np.abs(values)))))
    return [key for key, field in (("stirring", "StirringRPM"), ("volume", "InitialVolume_ml"))
            if varies(field) or (key == "volume" and varies("AverageVolume_ml"))]


def power_effects(models, processes):
    powers = np.asarray([p.get("PowerRatio", 1.) for p in processes])
    varied = len(powers) > 1 and not np.allclose(powers, powers[0])
    reports = []
    for index, model in enumerate(models):
        table = np.asarray(model["paraset"], dtype=float).copy()
        old_active = table[:, 0] == 1
        fitted = np.asarray(model.get("para", []), dtype=float).reshape(-1)
        if fitted.size == old_active.sum():
            table[old_active, 1] = fitted
        eligible = []
        for rate, exponent, label in ((5, 11, "secondary nucleation"), (24, 27, "agglomeration")):
            if not varied:
                table[exponent, 0] = 0
            if not np.isfinite(table[rate, 1]):
                continue  # An explicitly disabled mechanism stays disabled.
            if varied and table[exponent, 2] < table[exponent, 3]:
                table[exponent, 0] = 1
                eligible.append(label)
        model["paraset"] = table
        if not np.array_equal(old_active, table[:, 0] == 1):
            model["para"] = table[table[:, 0] == 1, 1].copy()
            model.pop("covmat", None)
        reports.append({"model": index + 1, "power_varies": bool(varied), "estimated_power_effects": eligible,
                        "note": "Power-exponent estimation uses the candidate's existing bounds. Set non-degenerate exponent bounds to estimate a currently fixed effect."})
    return reports
