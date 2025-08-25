```
# Kalman Filter vs OLS on a single-regressor time series
# - We simulate data where the true coefficient beta_t changes over time.
# - Compare static OLS to a time-varying-parameter (TVP) Kalman filter.
#
# Notes for plotting (per instructions):
# * Using matplotlib (no seaborn)
# * Each chart has its own figure (no subplots)
# * No explicit color choices

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import isfinite

np.random.seed(42)

# -----------------------------
# 1) Simulate data with time-varying beta_t
# -----------------------------
T = 500
x = np.random.normal(0, 1.0, size=T)

beta_true = np.zeros(T)
# piecewise / smoothly varying beta
beta_true[:150] = 0.5
beta_true[150:350] = np.linspace(0.5, -0.8, 200)
beta_true[350:] = -0.2
# add a small AR(1)-like jitter to the last regime
rho, jitter_sigma = 0.8, 0.05
for t in range(351, T):
    beta_true[t] += rho * (beta_true[t-1] - (-0.2)) + np.random.normal(0, jitter_sigma)

sigma_eps = 0.3  # measurement noise std
eps = np.random.normal(0, sigma_eps, size=T)
y = beta_true * x + eps

# -----------------------------
# 2) OLS (static) baseline
# -----------------------------
# y = beta * x + e
# beta_hat = sum(x*y) / sum(x^2)
beta_ols = np.sum(x * y) / np.sum(x**2)
yhat_ols = beta_ols * x
mse_ols = np.mean((y - yhat_ols) ** 2)

# Estimate R (measurement noise variance) from OLS residuals as a starting point
resid_ols = y - yhat_ols
R_est = np.var(resid_ols)  # measurement noise variance

# -----------------------------
# 3) Kalman Filter for time-varying beta_t (random walk)
#    beta_t = beta_{t-1} + eta_t,  eta_t ~ N(0, Q)
#    y_t    = x_t * beta_t + eps_t, eps_t ~ N(0, R)
# -----------------------------
def kalman_filter_tvp_beta(x, y, Q, R, beta0=0.0, P0=10.0):
    T = len(y)
    beta_filt = np.zeros(T)
    P_filt = np.zeros(T)
    yhat_1ahead = np.zeros(T)  # one-step-ahead prediction using pre-update state
    innov_var = np.zeros(T)

    beta_prev = beta0
    P_prev = P0

    for t in range(T):
        # Predict step (random walk)
        beta_pred = beta_prev
        P_pred = P_prev + Q

        # One-step-ahead prediction for y_t (before seeing y_t)
        yhat_1ahead[t] = x[t] * beta_pred

        # Innovation variance S_t = H P_pred H' + R = x_t^2 * P_pred + R
        S_t = x[t] ** 2 * P_pred + R
        innov_var[t] = S_t

        # Kalman gain: K_t = P_pred H' S_t^{-1} = P_pred * x_t / S_t
        K_t = P_pred * x[t] / S_t

        # Update step
        innov = y[t] - x[t] * beta_pred
        beta_new = beta_pred + K_t * innov
        P_new = (1.0 - K_t * x[t]) * P_pred

        # Store
        beta_filt[t] = beta_new
        P_filt[t] = P_new

        # Next
        beta_prev = beta_new
        P_prev = P_new

    return beta_filt, P_filt, yhat_1ahead, innov_var

# Simple grid search for Q (process variance) while fixing R to R_est
Q_grid = np.logspace(-6, -1, 10)
results = []
best = {"Q": None, "mse": np.inf, "beta": None, "P": None, "yhat": None}

for Q in Q_grid:
    beta_filt, P_filt, yhat_1ahead, S = kalman_filter_tvp_beta(x, y, Q=Q, R=R_est, beta0=0.0, P0=10.0)
    mse = np.mean((y - yhat_1ahead) ** 2)  # 1-step-ahead MSE
    results.append((Q, mse))
    if isfinite(mse) and mse < best["mse"]:
        best.update({"Q": Q, "mse": mse, "beta": beta_filt, "P": P_filt, "yhat": yhat_1ahead})

Q_best = best["Q"]
beta_kf = best["beta"]
yhat_kf = best["yhat"]
mse_kf = best["mse"]

# -----------------------------
# 4) Metrics table
# -----------------------------
import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user

metrics_df = pd.DataFrame({
    "Model": ["OLS (static β)", "Kalman (TVP β)"],
    "Key setting": [f"—", f"Q={Q_best:.2e}, R≈Var(OLS resid)"],
    "1-step-ahead MSE": [mse_ols, mse_kf],
    "β (mean over time)": [beta_ols, float(np.mean(beta_kf))]
})
display_dataframe_to_user("OLS vs Kalman (metrics)", metrics_df.round(6))

# -----------------------------
# 5) Plots
# -----------------------------

# (a) True β_t vs. filtered β_t and OLS β
plt.figure(figsize=(10, 4))
plt.plot(beta_true, label="True β_t")
plt.plot(beta_kf, label="Kalman filtered β_t", alpha=0.9)
plt.axhline(beta_ols, linestyle="--", label="OLS β (constant)")
plt.title("Time-varying β: True vs Kalman vs OLS")
plt.xlabel("t")
plt.ylabel("β")
plt.legend()
plt.tight_layout()
plt.show()

# (b) One-step-ahead predictions: y_t vs models' predictions
plt.figure(figsize=(10, 4))
plt.plot(y, label="y (observed)")
plt.plot(yhat_kf, label="Kalman 1-step-ahead ŷ_t", alpha=0.9)
plt.plot(yhat_ols, label="OLS in-sample ŷ", linestyle="--")
plt.title("Observed y vs model predictions")
plt.xlabel("t")
plt.ylabel("value")
plt.legend()
plt.tight_layout()
plt.show()

# (c) Grid search curve: Q vs 1-step-ahead MSE (Kalman)
Qs, MSEs = zip(*results)
plt.figure(figsize=(8, 4))
plt.plot(Qs, MSEs, marker="o")
plt.xscale("log")
plt.title("Kalman process variance (Q) vs 1-step-ahead MSE")
plt.xlabel("Q (log scale)")
plt.ylabel("MSE")
plt.tight_layout()
plt.show()

# Return a short textual summary so the user has context inline too.
summary = {
    "beta_ols": beta_ols,
    "Q_best": Q_best,
    "mse_ols": mse_ols,
    "mse_kf": mse_kf
}
summary

```



```
# Constant-beta case: OLS vs Kalman (counterexample where OLS should win)
# Requirements: matplotlib only, no seaborn; single-plot figures; no explicit colors.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import isfinite
from caas_jupyter_tools import display_dataframe_to_user

np.random.seed(123)

# -----------------------------
# 1) Simulate data with CONSTANT beta
# -----------------------------
T = 500
x = np.random.normal(0, 1.0, size=T)
beta_true = 0.6 * np.ones(T)   # constant coefficient
sigma_eps = 0.3
eps = np.random.normal(0, sigma_eps, size=T)
y = beta_true * x + eps

# -----------------------------
# 2) OLS (static) baseline
# -----------------------------
beta_ols = np.sum(x * y) / np.sum(x**2)
yhat_ols = beta_ols * x
mse_ols = np.mean((y - yhat_ols) ** 2)
R_est = np.var(y - yhat_ols)

# -----------------------------
# 3) Kalman Filter (random-walk beta) + grid over Q
# -----------------------------
def kalman_filter_tvp_beta(x, y, Q, R, beta0=0.0, P0=10.0):
    T = len(y)
    beta_filt = np.zeros(T)
    P_filt = np.zeros(T)
    yhat_1ahead = np.zeros(T)

    beta_prev = beta0
    P_prev = P0

    for t in range(T):
        # Predict
        beta_pred = beta_prev
        P_pred = P_prev + Q
        yhat_1ahead[t] = x[t] * beta_pred
        S_t = x[t] ** 2 * P_pred + R
        K_t = P_pred * x[t] / S_t
        innov = y[t] - x[t] * beta_pred
        beta_new = beta_pred + K_t * innov
        P_new = (1.0 - K_t * x[t]) * P_pred
        beta_filt[t] = beta_new
        P_filt[t] = P_new
        beta_prev = beta_new
        P_prev = P_new
    return beta_filt, P_filt, yhat_1ahead

Q_grid = np.logspace(-6, -1, 12)
best = {"Q": None, "mse": np.inf, "beta": None, "yhat": None}
grid = []
for Q in Q_grid:
    beta_kf, P_kf, yhat_kf = kalman_filter_tvp_beta(x, y, Q=Q, R=R_est, beta0=0.0, P0=10.0)
    mse = np.mean((y - yhat_kf) ** 2)
    grid.append((Q, mse))
    if isfinite(mse) and mse < best["mse"]:
        best.update({"Q": Q, "mse": mse, "beta": beta_kf, "yhat": yhat_kf})

Q_best = best["Q"]
beta_kf = best["beta"]
yhat_kf = best["yhat"]
mse_kf = best["mse"]

# -----------------------------
# 4) Metrics table
# -----------------------------
metrics_df = pd.DataFrame({
    "Model": ["OLS (static β)", "Kalman (TVP β)"],
    "Key setting": ["—", f"Q={Q_best:.2e}, R≈Var(OLS resid)"],
    "1-step-ahead MSE": [mse_ols, mse_kf],
    "β_OLS": [beta_ols, np.nan],
    "β_true": [beta_true[0], beta_true[0]],
    "mean(β_KF)": [np.nan, float(np.mean(beta_kf))]
}).round(6)
display_dataframe_to_user("Constant-beta case: metrics", metrics_df)

# -----------------------------
# 5) Plots
# -----------------------------

# (a) β trajectories
plt.figure(figsize=(10, 4))
plt.plot(beta_true, label="True β (constant)")
plt.plot(beta_kf, label="Kalman filtered β_t", alpha=0.9)
plt.axhline(beta_ols, linestyle="--", label="OLS β (constant, estimate)")
plt.title("Constant β case: True vs Kalman vs OLS")
plt.xlabel("t")
plt.ylabel("β")
plt.legend()
plt.tight_layout()
plt.show()

# (b) Predictions
plt.figure(figsize=(10, 4))
plt.plot(y, label="y (observed)")
plt.plot(yhat_kf, label="Kalman 1-step-ahead ŷ_t", alpha=0.9)
plt.plot(yhat_ols, label="OLS in-sample ŷ", linestyle="--")
plt.title("Observed y vs predictions (constant β)")
plt.xlabel("t")
plt.ylabel("value")
plt.legend()
plt.tight_layout()
plt.show()

# (c) Q-grid curve
Qs, MSEs = zip(*grid)
plt.figure(figsize=(8, 4))
plt.plot(Qs, MSEs, marker="o")
plt.xscale("log")
plt.title("Kalman process variance (Q) vs 1-step-ahead MSE (constant β)")
plt.xlabel("Q (log scale)")
plt.ylabel("MSE")
plt.tight_layout()
plt.show()

{"beta_ols": float(beta_ols), "Q_best": float(Q_best), "mse_ols": float(mse_ols), "mse_kf": float(mse_kf)}

```





```
```

