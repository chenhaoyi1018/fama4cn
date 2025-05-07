import baostock as bs
import pandas as pd
import numpy as np
import statsmodels.api as sm
import argparse
import sys
import os
import datetime

# 登录 Baostock
lg = bs.login()

def fetch_monthly_data(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 Baostock 获取单只 A 股的日线数据并汇总为月度：
      - return: 月度收益率 = last_close/first_close - 1
      - turnover_rate: 月度平均换手率
      - amount: 月度累计成交额
      - EP: 月度末滚动市盈率反比
    ts_code: Baostock 代码格式 e.g. "sh.600000"
    start_date, end_date: "YYYY-MM-DD"
    """
    # 只请求必要字段
    fields = "date,code,close,turn,amount,peTTM"
    rs = bs.query_history_k_data_plus(
        ts_code, fields,
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="2"
    )
    data = rs.get_data()
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    # 转换类型
    df['close'] = df['close'].astype(float)
    # Safely parse turnover rates, coercing invalid entries to NaN
    df['turnover_rate'] = pd.to_numeric(df['turn'], errors='coerce')
    df['amount'] = df['amount'].astype(float)
    df['peTTM'] = pd.to_numeric(df['peTTM'], errors='coerce')
    # 月度汇总
    monthly = pd.DataFrame({
        'return': df['close'].resample('M').last() / df['close'].resample('M').first() - 1,
        'turnover_rate': df['turnover_rate'].resample('M').mean(),
        'amount': df['amount'].resample('M').sum(),
        'EP': 1.0 / df['peTTM'].resample('M').last()
    })
    monthly['ts_code'] = ts_code
    monthly = monthly.dropna(subset=['return','EP'])
    return monthly.reset_index()

def compute_four_factors(
    ts_codes: list[str],
    start_date: str,
    end_date: str,
    rf_series: pd.Series
) -> pd.DataFrame:
    """
    构造 CH-4 四因子收益序列：MKT, SMB, VMG, PMO
    """
    # 汇总所有股票的月度数据
    frames = []
    for code in ts_codes:
        md = fetch_monthly_data(code, start_date, end_date)
        frames.append(md)
    all_month = pd.concat(frames, ignore_index=True)
    # 按月计算因子
    factor_list = []
    for date, grp in all_month.groupby('date'):
        mdf = grp.set_index('ts_code')
        # 剔除底部30%最小“规模” (这里用月度成交额 amount 代替市值)
        cutoff = mdf['amount'].quantile(0.3)
        trimmed = mdf[mdf['amount'] > cutoff]
        # SMB: 小规模 – 大规模
        med_amt = trimmed['amount'].median()
        smb = trimmed[trimmed['amount'] < med_amt]['return'].mean() \
            - trimmed[trimmed['amount'] >= med_amt]['return'].mean()
        # VMG: 高 EP – 低 EP
        med_ep = trimmed['EP'].median()
        vmg = trimmed[trimmed['EP'] >= med_ep]['return'].mean() \
            - trimmed[trimmed['EP'] < med_ep]['return'].mean()
        # MKT: 全市场平均 – 无风险
        mkt = mdf['return'].mean() - rf_series.get(date, 0)
        # PMO: 上10%换手率 – 下10%
        top10 = mdf['turnover_rate'].quantile(0.9)
        bot10 = mdf['turnover_rate'].quantile(0.1)
        pmo = mdf[mdf['turnover_rate'] >= top10]['return'].mean() \
            - mdf[mdf['turnover_rate'] <= bot10]['return'].mean()
        factor_list.append({
            'date': date, 'MKT': mkt, 'SMB': smb,
            'VMG': vmg, 'PMO': pmo
        })
    factor_df = pd.DataFrame(factor_list).set_index('date')
    return factor_df

def compute_factor_exposures(
    portfolio_returns: pd.DataFrame,
    factor_df: pd.DataFrame
) -> pd.DataFrame:
    """
    时间序列回归计算β
    """
    exposures = {}
    for port in portfolio_returns.columns:
        df = pd.concat([portfolio_returns[port], factor_df], axis=1, join='inner').dropna()
        Y = df[port]
        X = sm.add_constant(df[factor_df.columns])
        res = sm.OLS(Y, X).fit()
        exposures[port] = res.params[factor_df.columns]
    return pd.DataFrame(exposures).T

def fama_macbeth(
    portfolio_returns: pd.DataFrame,
    exposures_df: pd.DataFrame
):
    """
    Fama–MacBeth 交叉截面回归
    """
    rets = portfolio_returns[exposures_df.index]
    gamma_list = []
    for date in rets.index:
        y = rets.loc[date].dropna()
        X = exposures_df.loc[y.index]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        gamma_list.append(model.params)
    gamma_ts = pd.DataFrame(gamma_list, index=rets.index)
    gamma_mean = gamma_ts.mean()
    gamma_se = gamma_ts.std(ddof=1) / np.sqrt(len(gamma_ts))
    return gamma_ts, gamma_mean, gamma_se

def backup_daily_data(ts_codes: list[str], start_date: str, end_date: str, save_dir: str):
    """
    Download full daily history for each ts_code from Baostock and save to CSV.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Fields to retrieve
    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,peTTM"
    for code in ts_codes:
        print(f"Downloading {code} from {start_date} to {end_date}...")
        rs = bs.query_history_k_data_plus(
            code, fields,
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2"
        )
        data = rs.get_data()
        df = pd.DataFrame(data)
        # Save raw CSV
        filename = f"{code}_daily_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
        filepath = os.path.join(save_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved data to {filepath}")

if __name__ == "__main__":
    # 您的 A 股列表（Baostock 代码格式）
    ts_codes = ["sh.600004","sh.600007"]
    # 无风险率示例：3%年化拆分到月度
    rf_series = pd.Series(0.03/12, index=pd.date_range("2000-01-31", "2016-12-31", freq="M"))

    factor_df = compute_four_factors(ts_codes, "2000-01-01", "2016-12-31", rf_series)
    print("四因子收益率：\n", factor_df.head())

    # Parse command-line arguments for portfolio returns file
    parser = argparse.ArgumentParser(description="Compute CH-4 factor exposures and Fama-MacBeth regression")
    parser.add_argument("--portfolio", "-p", default="portfolio_returns.csv",
                        help="Path to portfolio returns CSV (first column is date, subsequent columns are portfolio returns)")
    parser.add_argument("--backup", action="store_true",
                        help="Download and save daily data for ts_codes")
    parser.add_argument("--backup-dir", default="backup_data",
                        help="Directory to save backup CSV files")
    args = parser.parse_args()

    if args.backup:
        # Backup daily data and exit
        today = datetime.date.today().strftime("%Y-%m-%d")
        backup_daily_data(ts_codes, "2000-01-01", today, args.backup_dir)
        bs.logout()
        sys.exit(0)

    portfolio_path = args.portfolio
    if not os.path.isfile(portfolio_path):
        print(f"Error: Portfolio returns file not found: {portfolio_path}", file=sys.stderr)
        sys.exit(1)

    portfolio_returns = pd.read_csv(portfolio_path, index_col=0, parse_dates=True)

    exposures_df = compute_factor_exposures(portfolio_returns, factor_df)
    print("\n组合因子暴露 β：\n", exposures_df)

    gamma_ts, gamma_mean, gamma_se = fama_macbeth(portfolio_returns, exposures_df)
    print("\n每月风险溢价：\n", gamma_ts.head())
    print("\n平均风险溢价：\n", gamma_mean)
    print("\n标准误：\n", gamma_se)

# 登出 Baostock
bs.logout()