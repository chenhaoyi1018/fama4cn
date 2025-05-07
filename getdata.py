import baostock as bs
import pandas as pd
import os
import datetime
import time
def fetch_profit_ttm(ts_code: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetch quarterly net profit TTM (epsTTM) from Q1 2007 onwards and forward-fill to daily index.
    Returns a daily Series indexed by date with name 'netProfitTTM'.
    """
    # Baostock profit_data (epsTTM) available from 2007 Q1 onwards
    raw_start_year = int(start_date[:4])
    start_year = max(raw_start_year, 2007)
    end_year = int(end_date[:4])
    profit_records = []
    # Fetch quarterly profit TTM for each quarter in the range
    for year in range(start_year, end_year + 1):
        for q in ["1", "2", "3", "4"]:
            rs = bs.query_profit_data(ts_code, str(year), q)
            while rs.error_code == '0' and rs.next():
                profit_records.append(rs.get_row_data())
    # If no profit data, return empty Series
    if not profit_records:
        return pd.Series([], name='netProfitTTM')
    dfp = pd.DataFrame(profit_records, columns=rs.fields)
    # Parse report dates and profit
    dfp['statDate'] = pd.to_datetime(dfp['statDate'])
    dfp['netProfitTTM'] = pd.to_numeric(dfp['netProfitTTM'], errors='coerce')
    dfp = dfp[['statDate', 'netProfitTTM']].set_index('statDate').sort_index()
    # Forward-fill to daily index
    daily_profit = dfp.resample('D').ffill()
    return daily_profit['netProfitTTM']

def get_all_ashares():
    """
    Retrieve all A-share codes currently listed in Shanghai and Shenzhen markets.
    """
    rs = bs.query_stock_basic(code="", code_name="")
    data_list = []
    # Iterate through the result set
    while rs.error_code == '0' and rs.next():
        data_list.append(rs.get_row_data())
    df_all = pd.DataFrame(data_list, columns=rs.fields)
    return df_all['code'].tolist()

def fetch_daily_indicators(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily indicators for a single A-share between start_date and end_date.
    Returns a DataFrame with columns:
    date, code, open, high, low, close, preclose, volume, amount,
    adjustflag, turn, tradestatus, pctChg, isST, peTTM
    """
    fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
    rs = bs.query_history_k_data_plus(
        ts_code, fields,
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="2"
    )
    data = rs.get_data()

    df = pd.DataFrame(data)
    # Skip if no data returned
    if df.empty or 'date' not in df.columns:
        print(f"Warning: no daily data for {ts_code}, skipping...")
        return pd.DataFrame()

    # Fetch epsTTM via quarterly profit_data and forward-fill to daily
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    eps_records = []
    for year in range(start_year, end_year + 1):
        for q in ["1", "2", "3", "4"]:
            rs_eps = bs.query_profit_data(ts_code, str(year), q)
            while rs_eps.error_code == '0' and rs_eps.next():
                eps_records.append(rs_eps.get_row_data())
    if eps_records:
        dfp = pd.DataFrame(eps_records, columns=rs_eps.fields)
        dfp['statDate'] = pd.to_datetime(dfp['statDate'])
        # Convert epsTTM field and forward-fill
        dfp['epsTTM'] = pd.to_numeric(dfp.get('epsTTM', pd.Series()), errors='coerce')
        dfp = dfp[['statDate','epsTTM']].set_index('statDate').sort_index()
        eps_daily = dfp['epsTTM'].resample('D').ffill()
    else:
        eps_daily = pd.Series(dtype=float, name='epsTTM')

    # Convert types for all fields
    df['date'] = pd.to_datetime(df['date'])
    for col in ['open','high','low','close','preclose','volume','amount','adjustflag','turn','tradestatus','pctChg','isST']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.set_index('date')
    # Join with epsTTM (forward-filled daily)
    df = df.join(eps_daily.rename('epsTTM'), how='left')
    # Compute price-to-earnings ratio (peTTM) = close / epsTTM
    df['peTTM'] = df['close'] / df['epsTTM']
    df = df.reset_index()
    return df[['date', 'code', 'open','high','low','close','preclose','volume','amount','adjustflag','turn','tradestatus','pctChg','isST','epsTTM','peTTM']]

def main(save_dir: str = "daily_data", start_date: str = "2008-01-01", end_date: str = None):
    """
    Download daily indicators for all A-shares and save each to CSV in save_dir.
    """
    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Identify already processed codes to resume
    existing_files = os.listdir(save_dir)
    processed = {fn.split('_')[0] for fn in existing_files if fn.endswith('.csv')}
    # Login to baostock
    lg = bs.login()
    codes = get_all_ashares()
    # Correct any Shenzhen codes mis-prefixed as Shanghai (sh.0xxxx) to proper sz.0xxxx
    codes = [code.replace('sh.', 'sz.', 1) if code.startswith('sh.0') else code for code in codes]
    # Only include standard A-shares: Shanghai main board (sh.6xxxx) and Shenzhen main board (sz.0xxxx)
    codes = [code for code in codes if code.startswith('sh.6') or code.startswith('sz.0')]
    # Skip codes already processed
    codes = [code for code in codes if code not in processed]
    for code in codes:
        try:
            print(f"Fetching {code} from {start_date} to {end_date}...")
            df = fetch_daily_indicators(code, start_date, end_date)
            if df.empty:
                continue
            filename = f"{code}_{start_date.replace('-','')}_{end_date.replace('-','')}.csv"
            filepath = os.path.join(save_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"Saved to {filepath}")
            # Throttle loop to avoid overloading the API
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching {code}: {e}. Continuing to next code.")
            continue
    bs.logout()

if __name__ == "__main__":
    main()