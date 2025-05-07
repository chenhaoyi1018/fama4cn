

import pandas as pd
import numpy as np
import argparse

def compute_pure_factor_returns(df: pd.DataFrame,
                                factor_cols: list,
                                ret_col: str = 'return',
                                top_q: float = 0.3,
                                bottom_q: float = 0.3) -> pd.DataFrame:
    """
    Construct pure factor long-short portfolios for each factor and compute time-series of portfolio returns.
    
    Parameters:
    - df: DataFrame containing at least ['date', factor_cols..., ret_col].
    - factor_cols: list of factor column names.
    - ret_col: name of the column with next-period returns.
    - top_q: fraction for long portfolio (e.g., 0.3 = top 30%).
    - bottom_q: fraction for short portfolio (e.g., 0.3 = bottom 30%).

    Returns:
    - DataFrame indexed by date, columns=factor_cols, each entry is long-minus-short return.
    """
    # Ensure date is datetime and sorted
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    results = {}
    for factor in factor_cols:
        def calc_return(group):
            # drop missing
            g = group.dropna(subset=[factor, ret_col])
            if g.empty:
                return np.nan
            top_cut = g[factor].quantile(1 - top_q)
            bot_cut = g[factor].quantile(bottom_q)
            long_ret = g.loc[g[factor] >= top_cut, ret_col].mean()
            short_ret = g.loc[g[factor] <= bot_cut, ret_col].mean()
            return long_ret - short_ret
        
        # compute per-date factor return
        fr = df.groupby('date').apply(calc_return)
        fr.name = factor
        results[factor] = fr
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute pure long-short factor portfolio returns")
    parser.add_argument("--input", "-i", required=True,
                        help="CSV file with columns date, code, factors, return")
    parser.add_argument("--factors", "-f", nargs="+", required=True,
                        help="List of factor column names, e.g. gk_rev obv_rev ...")
    parser.add_argument("--top-q", type=float, default=0.3,
                        help="Top quantile fraction for longs (default=0.3)")
    parser.add_argument("--bottom-q", type=float, default=0.3,
                        help="Bottom quantile fraction for shorts (default=0.3)")
    parser.add_argument("--output", "-o", default="pure_factor_returns.csv",
                        help="Output CSV for factor portfolio returns")
    args = parser.parse_args()

    data = pd.read_csv(args.input, parse_dates=['date'])
    fac_rets = compute_pure_factor_returns(data,
                                           factor_cols=args.factors,
                                           ret_col='return',
                                           top_q=args.top_q,
                                           bottom_q=args.bottom_q)
    fac_rets.to_csv(args.output, index=True)
    print(f"Pure factor portfolio returns saved to {args.output}")