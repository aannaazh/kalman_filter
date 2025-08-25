
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import isfinite

__all__ = ["kalman_filter_tvp_beta", "compare_ols_kalman", "demo_compare"]

def kalman_filter_tvp_beta(x, y, Q, R, beta0=0.0, P0=10.0):
    """Time-varying-parameter (random-walk) Kalman filter for a single regressor.

    State:     beta_t = beta_{t-1} + eta_t,   eta_t ~ N(0, Q)
    Measurement: y_t = x_t * beta_t + eps_t,  eps_t ~ N(0, R)

    Returns
    -------
    beta_filt : np.ndarray (T,)
        Filtered beta estimates.
    P_filt    : np.ndarray (T,)
        Posterior variance of beta.
    yhat_1ahead : np.ndarray (T,)
        One-step-ahead predictions for y before seeing y_t.
    S_t : np.ndarray (T,)
        Innovation variance at each step.
    """
    T = len(y)
    beta_filt = np.zeros(T)
    P_filt = np.zeros(T)
    yhat_1ahead = np.zeros(T)
    innov_var = np.zeros(T)

    beta_prev = beta0
    P_prev = P0

    for t in range(T):
        # Predict
        beta_pred = beta_prev
        P_pred = P_prev + Q

        yhat_1ahead[t] = x[t] * beta_pred
        S_t = x[t] ** 2 * P_pred + R
        innov_var[t] = S_t

        # Kalman gain
        K_t = P_pred * x[t] / S_t

        # Update
        innov = y[t] - x[t] * beta_pred
        beta_new = beta_pred + K_t * innov
        P_new = (1.0 - K_t * x[t]) * P_pred

        beta_filt[t] = beta_new
        P_filt[t] = P_new
        beta_prev = beta_new
        P_prev = P_new

    return beta_filt, P_filt, yhat_1ahead, innov_var


def _ols_fit(x, y):
    beta_ols = np.sum(x * y) / np.sum(x**2)
    yhat = beta_ols * x
    resid = y - yhat
    mse = np.mean(resid**2)
    return beta_ols, yhat, resid, mse


def compare_ols_kalman(x, y, Q_grid=None, R_mode="ols_resid", beta0=0.0, P0=10.0, plot=True):
    """Compare OLS vs Kalman (TVP-β) on given series x, y.

    Parameters
    ----------
    x, y : array-like
        1D arrays of equal length.
    Q_grid : array-like or None
        Candidate process variances for the random-walk β. If None, uses logspace 1e-6..1e-1.
    R_mode : {"ols_resid", float}
        If "ols_resid", set R = Var(OLS residuals). Or pass a numeric R (variance).
    beta0 : float
        Prior mean for β_0.
    P0 : float
        Prior variance for β_0.
    plot : bool
        If True, generate 3 figures: β-trajectory, y predictions, Q-vs-MSE.

    Returns
    -------
    result : dict
        {
          "beta_ols": float,
          "mse_ols": float,
          "Q_best": float,
          "mse_kf": float,
          "beta_kf": np.ndarray,
          "yhat_kf": np.ndarray,
          "yhat_ols": np.ndarray,
          "metrics_df": pd.DataFrame,
          "grid": list of (Q, mse)
        }
    """
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    assert x.ndim == y.ndim == 1 and x.shape[0] == y.shape[0], "x and y must be 1D arrays with equal length"
    T = len(y)

    # OLS
    beta_ols, yhat_ols, resid_ols, mse_ols = _ols_fit(x, y)

    # Measurement noise variance R
    if R_mode == "ols_resid":
        R = float(np.var(resid_ols))
    else:
        R = float(R_mode)

    # Q grid
    if Q_grid is None:
        Q_grid = np.logspace(-6, -1, 10)

    best = {"Q": None, "mse": np.inf, "beta": None, "P": None, "yhat": None}
    grid = []

    for Q in Q_grid:
        beta_filt, P_filt, yhat_1ahead, S = kalman_filter_tvp_beta(x, y, Q=Q, R=R, beta0=beta0, P0=P0)
        mse = float(np.mean((y - yhat_1ahead)**2))
        grid.append((float(Q), mse))
        if isfinite(mse) and mse < best["mse"]:
            best.update({"Q": float(Q), "mse": mse, "beta": beta_filt, "P": P_filt, "yhat": yhat_1ahead})

    # Prepare metrics df
    metrics_df = pd.DataFrame({
        "Model": ["OLS (static β)", "Kalman (TVP β)"],
        "Key setting": ["—", f"Q={best['Q']:.2e}, R≈Var(OLS resid)" if R_mode=="ols_resid" else f"Q={best['Q']:.2e}, R={R:.3g}"],
        "1-step-ahead MSE": [mse_ols, best["mse"]],
        "β_OLS": [beta_ols, np.nan],
        "mean(β_KF)": [np.nan, float(np.mean(best["beta"]))]
    })

    # Plots
    if plot:
        # A) β trajectories
        plt.figure(figsize=(10, 4))
        plt.plot(best["beta"], label="Kalman filtered β_t")
        plt.axhline(beta_ols, linestyle="--", label="OLS β (constant)")
        plt.title("β trajectory: Kalman vs OLS constant β")
        plt.xlabel("t")
        plt.ylabel("β")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # B) Predictions
        plt.figure(figsize=(10, 4))
        plt.plot(y, label="y (observed)")
        plt.plot(best["yhat"], label="Kalman 1-step-ahead ŷ_t")
        plt.plot(yhat_ols, label="OLS in-sample ŷ", linestyle="--")
        plt.title("Observed y vs model predictions")
        plt.xlabel("t")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # C) Q vs MSE
        Qs, MSEs = zip(*grid)
        plt.figure(figsize=(8, 4))
        plt.plot(Qs, MSEs, marker="o")
        plt.xscale("log")
        plt.title("Kalman process variance (Q) vs 1-step-ahead MSE")
        plt.xlabel("Q (log scale)")
        plt.ylabel("MSE")
        plt.tight_layout()
        plt.show()

    result = {
        "beta_ols": float(beta_ols),
        "mse_ols": float(mse_ols),
        "Q_best": float(best["Q"]),
        "mse_kf": float(best["mse"]),
        "beta_kf": best["beta"],
        "yhat_kf": best["yhat"],
        "yhat_ols": yhat_ols,
        "metrics_df": metrics_df,
        "grid": grid,
    }
    return result


def demo_compare(T=400, regime="timevary", seed=42):
    """Generate a simple demo and run compare_ols_kalman.

    regime: "timevary" or "constant"
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=T)
    if regime == "timevary":
        beta_true = np.zeros(T)
        beta_true[:T//3] = 0.5
        mid = T//3
        end = 2*T//3
        beta_true[mid:end] = np.linspace(0.5, -0.8, end-mid)
        beta_true[end:] = -0.2
        y = beta_true * x + rng.normal(0.0, 0.3, size=T)
    else:
        beta_true = np.full(T, 0.6)
        y = beta_true * x + rng.normal(0.0, 0.3, size=T)

    res = compare_ols_kalman(x, y, plot=True)
    return {
        "regime": regime,
        "beta_ols": res["beta_ols"],
        "mse_ols": res["mse_ols"],
        "Q_best": res["Q_best"],
        "mse_kf": res["mse_kf"],
    }
