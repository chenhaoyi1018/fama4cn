import pandas as pd
import numpy as np
import argparse
import os
import gc

# Factor calculation functions (only reversed factors included)
def factor_obv(df):
    sign = np.sign(df['close'].diff().fillna(0))
    obv = (sign * df['volume']).groupby(df['ts_code']).cumsum()
    return obv

def factor_gk_vol(df, window=14):
    hl = np.log(df['high'] / df['low'])
    co = np.log(df['close'] / df['open'])
    gk = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    avg = gk.groupby(df['ts_code']).transform(lambda x: x.rolling(window).mean())
    return np.sqrt(avg)

def factor_ret_vol_corr(df, window=20):
    ret = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change(fill_method=None))
    vol = df['volume']
    tmp = pd.DataFrame({'ret': ret, 'vol': vol})
    corr = tmp.groupby(df['ts_code']).apply(
        lambda x: x['ret'].rolling(window).corr(x['vol'])
    ).reset_index(level=0, drop=True)
    return corr

def factor_resid_vol(df, window=21):
    # Requires market return
    ret = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change(fill_method=None))
    ret_mkt = ret.groupby(df['date']).transform('mean')
    beta = ret.groupby(df['ts_code']).rolling(window).cov(ret_mkt).reset_index(level=0, drop=True) / ret_mkt.groupby(df['ts_code']).rolling(window).var().reset_index(level=0, drop=True)
    resid = ret - beta * ret_mkt
    return resid.groupby(df['ts_code']).transform(lambda x: x.rolling(window).std() * np.sqrt(252))

def factor_rel_mom(df, window=21):
    mom = df.groupby('ts_code')['close'].transform(lambda x: x.pct_change(window, fill_method=None))
    med = mom.groupby(df['date']).transform('median')
    return mom - med

def factor_slope(df, window=21):
    from scipy.stats import linregress
    def slope(x):
        idx = np.arange(len(x))
        return linregress(idx, x).slope
    return df.groupby('ts_code')['close'].transform(lambda x: x.rolling(window).apply(slope, raw=True))

def factor_ema_diff(df, short=10, long=50):
    ema_s = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=short, adjust=False).mean())
    ema_l = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=long, adjust=False).mean())
    return ema_s - ema_l

# Reversed factor wrappers
def factor_obv_rev(df): return -factor_obv(df)
def factor_gk_vol_rev(df): return -factor_gk_vol(df)
def factor_ret_vol_corr_20_rev(df): return -factor_ret_vol_corr(df, 20)
def factor_resid_vol_21_rev(df): return -factor_resid_vol(df, 21)
def factor_rel_mom_21_rev(df): return -factor_rel_mom(df, 21)
def factor_slope_21_rev(df): return -factor_slope(df, 21)
def factor_ema_diff_rev(df): return -factor_ema_diff(df, 10, 50)

def compute_pure_factor_returns(df, factor_cols, ret_col='return', top_q=0.3, bottom_q=0.3):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    results = {}
    for factor in factor_cols:
        def calc_return(group):
            g = group.dropna(subset=[factor, ret_col])
            if g.empty: return np.nan
            # Delay factor and ret_col to next trading day
            g['next_day_factor'] = g[factor].shift(-1)
            g['next_day_ret'] = g[ret_col].shift(-1)
            top_cut = g['next_day_factor'].quantile(1 - top_q)
            bot_cut = g['next_day_factor'].quantile(bottom_q)
            long_ret = g.loc[g['next_day_factor'] >= top_cut, 'next_day_ret'].mean()
            short_ret = g.loc[g['next_day_factor'] <= bot_cut, 'next_day_ret'].mean()
            return long_ret - short_ret
        fr = df.groupby('date').apply(calc_return)
        fr.name = factor
        results[factor] = fr
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calc reversed factors and pure portfolio returns")
    parser.add_argument(
        "-d", "--data-dir",
        default="daily_data",
        help="Directory containing per-stock CSV files (filename should start with ts_code)"
    )
    parser.add_argument("-o", "--output", default="pure_factor_returns.csv",
                        help="Output CSV for factor portfolio returns")
    parser.add_argument("--top-q", type=float, default=0.3)
    parser.add_argument("--bottom-q", type=float, default=0.3)
    args = parser.parse_args()

    # Factor function mapping
    factor_funcs = {
        'obv_rev': factor_obv_rev,
        'gk_vol_rev': factor_gk_vol_rev,
        'ret_vol_corr_20_rev': factor_ret_vol_corr_20_rev,
        'resid_vol_21_rev': factor_resid_vol_21_rev,
        'rel_mom_21_rev': factor_rel_mom_21_rev,
        'slope_21_rev': factor_slope_21_rev,
        'ema_diff_rev': factor_ema_diff_rev
    }

    # Load all per-stock CSVs into one DataFrame with memory optimizations
    frames = []
    for filename in os.listdir(args.data_dir):
        if not filename.endswith('.csv'):
            continue
        ts_code = filename.split('_')[0]
        path = os.path.join(args.data_dir, filename)
        # Load only necessary columns with optimized dtypes
        df_stock = pd.read_csv(
            path,
            usecols=['date','open','high','low','close','volume'],
            parse_dates=['date'],
            dtype={
                'open':'float32','high':'float32','low':'float32',
                'close':'float32','volume':'float32'
            }
        )
        df_stock['ts_code'] = ts_code
        # Convert ts_code to category to save memory
        df_stock['ts_code'] = df_stock['ts_code'].astype('category')
        # Compute daily return with no fill
        df_stock['return'] = df_stock.groupby('ts_code')['close'].pct_change(fill_method=None).astype('float32')
        # Compute reversed factors and cast to float32
        for name, func in factor_funcs.items():
            val = func(df_stock)
            # If the factor returns a DataFrame, take the first column
            if isinstance(val, pd.DataFrame):
                val = val.iloc[:, 0]
            df_stock[name] = val.astype('float32')
        # Keep only minimal columns
        keep_cols = ['date','ts_code','return'] + list(factor_funcs.keys())
        frames.append(df_stock[keep_cols])
        # Drop raw DataFrame and collect garbage
        del df_stock
        gc.collect()
    if not frames:
        raise ValueError(f"No CSV files found in {args.data_dir}")
    df = pd.concat(frames, ignore_index=True)
    # compute pure factor portfolio returns
    fac_rets = compute_pure_factor_returns(df, list(factor_funcs.keys()), ret_col='return', top_q=args.top_q, bottom_q=args.bottom_q)
    fac_rets.to_csv(args.output)
    print(f"Saved pure factor returns to {args.output}")