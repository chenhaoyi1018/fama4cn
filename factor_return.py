
import pandas as pd
import matplotlib.pyplot as plt

# 1. 加载数据：假设已有 gk_vol_rev 日收益率文件 pure_factor_returns.csv
df = pd.read_csv('pure_factor_returns.csv', index_col=0, parse_dates=True)

# 2. 计算累积收益率
cumulative_returns = (1 + df['gk_vol_rev']).cumprod()

# 3. 绘制累积收益率图
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns.index, cumulative_returns, label='gk_vol_rev Cumulative Returns', color='b', linewidth=2)
plt.title('Cumulative Returns of gk_vol_rev Factor')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()