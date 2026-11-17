"""Model screening from MATLAB IMED's run_parameter_estimation.m."""
import numpy as np
from scipy.stats import chi2, norm


def screen_models(fits, *, sample_count=None, delta_aic=None):
    """Apply the reference F and t checks and, for LOOP, the configured ΔAIC limit."""
    if not fits:
        return {}
    comparable = {index: fit for index, fit in fits.items()
                  if fit.converged and np.isfinite(fit.ssr) and fit.ssr >= 0}
    ranked = sorted((index for index in comparable if np.isfinite(fits[index].aic)),
                    key=lambda index: (fits[index].aic, index))
    ranks = {index: rank for rank, index in enumerate(ranked, 1)}
    baseline = min((fit.ssr for fit in comparable.values()), default=np.nan)
    # MATLAB uses finv(.95, number of candidates, Inf) and tinv(.975, Inf).
    f_limit = float(chi2.ppf(.95, len(fits)) / len(fits))
    t_limit = float(norm.ppf(.975))
    results = {}
    for index, fit in fits.items():
        ratio = np.nan
        if index in comparable:
            ratio = fit.ssr / baseline if baseline > 0 else (1. if fit.ssr == 0 else np.inf)
        diagonal = np.diag(fit.covariance)
        values = np.abs(np.asarray(fit.parameters).reshape(-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            statistics = values / np.sqrt(diagonal)
        # A nonfinite covariance cannot support a parameter significance decision.
        statistics[~np.isfinite(diagonal)] = np.nan
        f_pass = bool(ratio <= f_limit)
        t_pass = bool(statistics.size and (statistics >= t_limit).all())
        reasons = []
        if not fit.converged:
            reasons.append("fit did not converge")
        elif np.isnan(ratio):
            reasons.append("F-test unavailable")
        elif not f_pass:
            reasons.append("F-test failed")
        if np.isnan(statistics).any() or not statistics.size:
            reasons.append("parameter t-tests unavailable")
        elif not t_pass:
            reasons.append("parameter t-test failed")
        total_aic = float(fit.aic * sample_count) if index in ranks and sample_count is not None else None
        if delta_aic is not None and len(fits) > 1 and total_aic is None:
            reasons.append("AIC unavailable")
        results[index] = {"rank": ranks.get(index), "total_aic": total_aic, "delta_aic": None,
                          "f_value": float(ratio), "f_limit": f_limit, "f_pass": f_pass,
                          "t_statistics": statistics, "t_limit": t_limit, "t_pass": t_pass,
                          "reasons": reasons}
    if delta_aic is not None:
        best = min((result["total_aic"] for result in results.values()
                    if not result["reasons"] and result["total_aic"] is not None), default=None)
        if best is not None:
            for result in results.values():
                if result["total_aic"] is not None:
                    difference = result["total_aic"] - best
                    result["delta_aic"] = float(difference)
                    if difference > delta_aic:
                        result["reasons"].append(f"ΔAIC exceeds {delta_aic:g}")
    return results


def matlab_screening_fields(result):
    return {"r": result["rank"] if result["rank"] is not None else np.nan,
            "FVector": result["f_value"], "t_test": result["t_statistics"],
            "Results": int(not result["reasons"])}


def pack_screening(result):
    return {key: value if value is not None else np.nan for key, value in result.items()}


def unpack_screening(result):
    # MATLAB's reader squeezes scalar arrays and converts a one-entry reason list to a string.
    return {**result, "t_statistics": np.asarray(result["t_statistics"], dtype=float).reshape(-1),
            "rank": int(result["rank"]) if np.isfinite(result["rank"]) else None,
            "reasons": [reason.strip() for reason in np.asarray(result["reasons"], dtype=str).reshape(-1)]}
