
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import isfinite

from ols_kf_compare import compare_ols_kalman

__all__ = ["compare_ols_kalman_panel", "simulate_panel"]

def compare_ols_kalman_panel(df, date_col, symbol_col, x_col, y_col,
                             Q_grid=None, R_mode="ols_resid",
                             beta0=0.0, P0=10.0,
                             plot_samples=3, random_seed=42):
    """Group by symbol and run OLS vs Kalman (TVP-β) per symbol.

    Parameters
    ----------
    df : DataFrame with columns [date_col, symbol_col, x_col, y_col]
    Q_grid, R_mode, beta0, P0 : forwarded to single-series compare
    plot_samples : int
        Plot β_trajectory and predictions for a random subset of symbols.

    Returns
    -------
    panel_metrics : DataFrame
        Columns: symbol, n, mse_ols, mse_kf, rel_impr, abs_impr, Q_best, beta_ols, mean_beta_kf
    results_dict : dict
        symbol -> full single-series result dict
    """
    work = df[[date_col, symbol_col, x_col, y_col]].copy()
    work = work.dropna().sort_values([symbol_col, date_col])

    symbols = work[symbol_col].unique().tolist()
    rng = np.random.default_rng(random_seed)
    sample_syms = set(rng.choice(symbols, size=min(plot_samples, len(symbols)), replace=False))

    rows = []
    results = {}

    for sym, g in work.groupby(symbol_col, sort=False):
        x = g[x_col].to_numpy(dtype=float)
        y = g[y_col].to_numpy(dtype=float)
        res = compare_ols_kalman(x, y, Q_grid=Q_grid, R_mode=R_mode, beta0=beta0, P0=P0, plot=(sym in sample_syms))
        results[sym] = res
        n = len(g)
        abs_impr = res["mse_ols"] - res["mse_kf"]
        rel_impr = abs_impr / res["mse_ols"] if res["mse_ols"] > 0 else np.nan
        rows.append({
            "symbol": sym,
            "n": n,
            "mse_ols": res["mse_ols"],
            "mse_kf": res["mse_kf"],
            "abs_impr": abs_impr,
            "rel_impr": rel_impr,
            "Q_best": res["Q_best"],
            "beta_ols": res["beta_ols"],
            "mean_beta_kf": float(np.mean(res["beta_kf"])) if res["beta_kf"] is not None else np.nan,
        })

    panel_metrics = pd.DataFrame(rows).sort_values("abs_impr", ascending=False).reset_index(drop=True)

    # Distribution plot
    plt.figure(figsize=(8,4))
    plt.hist(panel_metrics["rel_impr"].dropna(), bins=20)
    plt.title("Distribution of relative MSE improvement (OLS → Kalman)")
    plt.xlabel("relative improvement")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    # Top-K improvement plot
    topK = panel_metrics.head(min(10, len(panel_metrics)))
    plt.figure(figsize=(10,4))
    plt.bar(topK["symbol"].astype(str), topK["abs_impr"])
    plt.title("Top symbols by absolute MSE improvement")
    plt.xlabel("symbol")
    plt.ylabel("abs_impr (MSE_OLS - MSE_KF)")
    plt.tight_layout()
    plt.show()

    return panel_metrics, results


def simulate_panel(T=300, M=24, frac_timevary=0.6, seed=123):
    """Simulate panel dataset with mixture of time-varying and constant-β symbols."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=T, freq="D")
    syms = [f"S{i:03d}" for i in range(M)]
    rows = []
    for i, s in enumerate(syms):
        regime = "timevary" if rng.random() < frac_timevary else "constant"
        x = rng.normal(0.0, 1.0, size=T)
        if regime == "timevary":
            beta = np.zeros(T)
            cut1 = T//3
            cut2 = 2*T//3
            start = 0.3 + 0.4 * rng.random()
            end = -0.7 + 0.4 * rng.random()
            beta[:cut1] = start
            beta[cut1:cut2] = np.linspace(start, end, cut2-cut1)
            beta[cut2:] = end + 0.1*rng.normal(size=T-cut2)
        else:
            const = 0.4 + 0.4 * rng.random()
            beta = np.full(T, const)
        eps = rng.normal(0.0, 0.35, size=T)
        y = beta * x + eps
        for t in range(T):
            rows.append((dates[t], s, x[t], y[t]))
    df = pd.DataFrame(rows, columns=["date", "symbol", "x", "y"])
    return df
