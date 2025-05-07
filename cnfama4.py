import os
import pandas as pd
import numpy as np
import argparse

def load_monthly_data(data_dir: str) -> pd.DataFrame:
    """
    从 data_dir 下所有 CSV 中读取日线数据，按月汇总：
      - return: 月度收益 = last_close/first_close - 1
      - size: 月度规模代理 = 月度累计成交额 (amount)
      - turn: 月度平均换手率
      - epsTTM: 月度末 epsTTM
    返回 DataFrame，columns=['date','code','return','size','turn','epsTTM']
    """
    records = []
    for fn in os.listdir(data_dir):
        if not fn.endswith('.csv'): continue
        code = fn.split('_')[0]
        df = pd.read_csv(os.path.join(data_dir, fn), parse_dates=['date'])
        df = df.sort_values('date').set_index('date')
        monthly = pd.DataFrame({
            'return' : df['close'].resample('M').last() / df['close'].resample('M').first() - 1,
            'size'   : df['amount'].resample('M').sum(),
            'turn'   : df['turn'].resample('M').mean(),
            'epsTTM' : df['epsTTM'].resample('M').last()
        })
        monthly['code'] = code
        monthly = monthly.dropna(subset=['return','size','turn','epsTTM'])
        records.append(monthly.reset_index())
    return pd.concat(records, ignore_index=True)

def compute_fama4cn(monthly: pd.DataFrame, rf: float) -> pd.DataFrame:
    """
    基于月度 DataFrame 计算 Fama4CN 因子序列：
      - MKT = 全市场平均 return - rf
      - SMB = 剔除底部30% size 后，小/大市值组合 return 差
      - VMG = 高/低 epsTTM 组合 return 差
      - PMO = 上10%/下10% turn 组合 return 差
    返回 DataFrame，index 为月末日期，columns=['MKT','SMB','VMG','PMO']
    """
    # Exclude bottom 30% by size helper
    def trim30(g):
        cutoff = g['size'].quantile(0.3)
        return g[g['size'] > cutoff]

    monthly['rf'] = rf
    # 市场因子
    mkt = (monthly
           .groupby('date')
           .apply(lambda g: trim30(g)['return'].mean() - rf)
           .rename('MKT'))
    # 规模因子
    def calc_smb(g):
        g2 = trim30(g)
        med = g2['size'].median()
        small = g2.loc[g2['size'] < med, 'return'].mean()
        big   = g2.loc[g2['size'] >= med, 'return'].mean()
        return small - big
    smb = monthly.groupby('date').apply(calc_smb).rename('SMB')
    # 价值因子
    def calc_vmg(g):
        g2 = trim30(g)
        med = g2['epsTTM'].median()
        high = g2.loc[g2['epsTTM'] >= med, 'return'].mean()
        low  = g2.loc[g2['epsTTM'] <  med, 'return'].mean()
        return high - low
    vmg = monthly.groupby('date').apply(calc_vmg).rename('VMG')
    # 情绪因子
    def calc_pmo(g):
        g2 = trim30(g)
        top = g2['turn'].quantile(0.9)
        bot = g2['turn'].quantile(0.1)
        return g2.loc[g2['turn'] >= top, 'return'].mean() - g2.loc[g2['turn'] <= bot, 'return'].mean()
    pmo = monthly.groupby('date').apply(calc_pmo).rename('PMO')
    # 合并
    factors = pd.concat([mkt, smb, vmg, pmo], axis=1)
    return factors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Fama4CN factors from daily CSVs")
    parser.add_argument("--data-dir", default="daily_data",
                        help="目录，包含各股日线 CSV")
    parser.add_argument("--rf", type=float, default=0.03/12,
                        help="月度无风险利率 (default 3%/12)")
    parser.add_argument("--out", default="fama4cn_factors.csv",
                        help="输出因子序列 CSV 文件名")
    args = parser.parse_args()

    print("加载并聚合月度数据...")
    monthly_df = load_monthly_data(args.data_dir)
    print("共计股票*月份 记录：", len(monthly_df))
    print("计算 Fama4CN 四因子...")
    fac = compute_fama4cn(monthly_df, rf=args.rf)
    fac.to_csv(args.out)
    print(f"完成！已保存因子序列到 {args.out}")