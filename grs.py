import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import argparse

def rolling_alpha_tstat(port_series: pd.Series, factor_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling Jensen's alpha of port_series against factor_df.
    Returns a DataFrame of alphas and their t-statistics indexed by the end date of each window.
    """
    alphas = []
    dates = []
    tstats = []
    for end in range(window - 1, len(port_series)):
        idx = port_series.index[end - window + 1: end + 1]
        # Align y and X, drop missing
        y = port_series.loc[idx]
        X = factor_df.loc[idx]
        data_win = pd.concat([y, X], axis=1).dropna()
        # Need at least as many observations as regressors (constant + factors)
        if data_win.shape[0] < (X.shape[1] + 1):
            continue
        y2 = data_win[y.name]
        X2 = sm.add_constant(data_win[X.columns])
        model = sm.OLS(y2, X2).fit()
        alphas.append(model.params.get('const', np.nan))
        tstats.append(model.tvalues.get('const', np.nan))
        dates.append(idx[-1])
    return pd.DataFrame({'alpha': alphas, 'tstat': tstats}, index=dates)

def main():
    parser = argparse.ArgumentParser(
        description="Plot rolling alphas for pure factor portfolios"
    )
    parser.add_argument(
        "--pure", "-p",
        default="pure_factor_returns.csv",
        help="CSV of pure factor portfolio returns (index=date, columns are factor names)"
    )
    parser.add_argument(
        "--factors", "-f",
        default="fama4cn_factors.csv",
        help="CSV of Fama4CN factor returns (index=date, columns=MKT,SMB,VMG,PMO)"
    )
    parser.add_argument(
        "--window", "-w",
        type=int, default=36,
        help="Rolling window size in months (default 36)"
    )
    args = parser.parse_args()

    # Load data
    pure = pd.read_csv(args.pure, index_col=0, parse_dates=True)
    factors = pd.read_csv(args.factors, index_col=0, parse_dates=True)

    # Determine common dates between pure returns and factors
    common_dates = pure.index.intersection(factors.index)

    # Plot rolling alphas for each portfolio
    for port in pure.columns:
        # Align series for this portfolio
        port_series = pure[port].reindex(common_dates)
        factor_df = factors.reindex(common_dates)
        result = rolling_alpha_tstat(
            port_series, factor_df, args.window
        )
        if result.empty:
            print(f"No rolling alpha data for {port}, skipping plot.")
            continue
        alpha_series = result['alpha']
        tstat_series = result['tstat']
        plt.figure(figsize=(10, 4))
        plt.plot(alpha_series.index, alpha_series.values, label=port)
        plt.title(f"Rolling {args.window}-Month Alpha for {port}")
        plt.xlabel("Date")
        plt.ylabel("Alpha")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        print(f"T-statistics for {port}:\n", tstat_series)

if __name__ == "__main__":
    main()