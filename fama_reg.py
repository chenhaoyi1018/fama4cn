import pandas as pd
import numpy as np
import statsmodels.api as sm
import argparse

def compute_factor_exposures(portfolio_returns: pd.DataFrame,
                             factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Time-series regress each portfolio's returns on factor returns to obtain exposures (betas).
    Returns a DataFrame of exposures: rows=portfolio names, columns=factor names.
    """
    exposures = {}
    for port in portfolio_returns.columns:
        # Align returns and factors
        df = pd.concat([portfolio_returns[port], factor_df], axis=1, join='inner').dropna()
        # Skip if there is no overlapping data
        if df.empty:
            exposures[port] = pd.Series(np.nan, index=factor_df.columns)
            continue
        Y = df[port]
        X = sm.add_constant(df[factor_df.columns])
        res = sm.OLS(Y, X).fit()
        # Store factor loadings (exclude constant)
        exposures[port] = res.params[factor_df.columns]
    return pd.DataFrame(exposures).T

def fama_macbeth(portfolio_returns: pd.DataFrame,
                 exposures_df: pd.DataFrame) -> tuple:
    """
    Perform Fama–MacBeth cross-sectional regressions:
    1) For each date, regress cross-section of portfolio returns on their exposures.
    Returns:
      - gamma_ts: DataFrame of risk premia time series (index=date)
      - gamma_mean: Series of average risk premia
      - gamma_se: Series of standard errors of average premia
    """
    # Align portfolios: only use common portfolio columns
    common_ports = exposures_df.index.intersection(portfolio_returns.columns)
    rets = portfolio_returns[common_ports]
    gamma_list = []
    dates = []
    for date in rets.index:
        # get returns at date and align with exposures
        y = rets.loc[date]
        X = exposures_df.loc[y.index]
        # combine and drop any portfolios with missing returns or exposures
        data_cs = pd.concat([y, X], axis=1, join='inner').dropna()
        y_cs = data_cs[y.name]
        X_cs = data_cs[X.columns]
        X_cs = sm.add_constant(X_cs)
        # if no data, skip
        if X_cs.empty:
            continue
        model = sm.OLS(y_cs, X_cs).fit()
        gamma_list.append(model.params)
        dates.append(date)
    # Use actual dates for which regressions were run
    gamma_ts = pd.DataFrame(gamma_list, index=dates)
    gamma_mean = gamma_ts.mean()
    gamma_se = gamma_ts.std(ddof=1) / np.sqrt(len(gamma_ts))
    return gamma_ts, gamma_mean, gamma_se

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fama–MacBeth regression for portfolio returns")
    parser.add_argument("--portfolios", "-p", default="pure_factor_returns.csv",
                        help="CSV of portfolio returns (index=date, columns=portfolio names)")
    parser.add_argument("--factors", "-f", default="fama4cn_factors.csv",
                        help="CSV of factor returns (index=date, columns=factor names)")
    parser.add_argument("--out-gamma-ts", default="gamma_ts.csv",
                        help="Output CSV for gamma time-series")
    parser.add_argument("--out-gamma-mean", default="gamma_mean.csv",
                        help="Output CSV for average gamma")
    parser.add_argument("--out-gamma-se", default="gamma_se.csv",
                        help="Output CSV for gamma standard errors")
    parser.add_argument("--out-gamma-tstat", default="gamma_tstat.csv",
                        help="Output CSV for gamma t-statistics")
    args = parser.parse_args()

    # Load data
    portfolios = pd.read_csv(args.portfolios, index_col=0, parse_dates=True)
    factors = pd.read_csv(args.factors, index_col=0, parse_dates=True)

    # Compute exposures
    exposures = compute_factor_exposures(portfolios, factors)

    # Run Fama–MacBeth
    gamma_ts, gamma_mean, gamma_se = fama_macbeth(portfolios, exposures)

    # Compute t-statistics: gamma_mean / gamma_se
    gamma_tstat = gamma_mean / gamma_se

    # Save results
    gamma_ts.to_csv(args.out_gamma_ts)
    gamma_mean.to_csv(args.out_gamma_mean, header=True)
    gamma_se.to_csv(args.out_gamma_se, header=True)
    gamma_tstat.to_csv(args.out_gamma_tstat, header=True)

    print("Fama–MacBeth regression complete. Results saved.")