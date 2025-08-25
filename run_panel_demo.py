
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ols_kf_compare import compare_ols_kalman
from ols_kf_panel import compare_ols_kalman_panel, simulate_panel

def main():
    parser = argparse.ArgumentParser(description="Run OLS vs Kalman (TVP-β) on a panel dataset.")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV file with columns: date,symbol,x,y")
    parser.add_argument("--date_col", type=str, default="date")
    parser.add_argument("--symbol_col", type=str, default="symbol")
    parser.add_argument("--x_col", type=str, default="x")
    parser.add_argument("--y_col", type=str, default="y")
    parser.add_argument("--out_csv", type=str, default="panel_metrics.csv")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--T", type=int, default=280)
    parser.add_argument("--M", type=int, default=30)
    parser.add_argument("--frac_timevary", type=float, default=0.65)
    parser.add_argument("--plot_samples", type=int, default=4)
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=[args.date_col])
    else:
        df = simulate_panel(T=args.T, M=args.M, frac_timevary=args.frac_timevary, seed=args.seed)

    panel_metrics, results = compare_ols_kalman_panel(
        df,
        date_col=args.date_col,
        symbol_col=args.symbol_col,
        x_col=args.x_col,
        y_col=args.y_col,
        plot_samples=args.plot_samples,
        random_seed=args.seed,
    )

    panel_metrics.to_csv(args.out_csv, index=False)
    print(f"Saved metrics to {args.out_csv}")
    print(panel_metrics.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
