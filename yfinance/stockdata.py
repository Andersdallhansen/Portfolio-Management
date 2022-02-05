import yfinance as yf
import pandas as pd



def get_data(stock_tickers, start, end, interval):
    data = yf.download(tickers=stock_tickers,start=start, end=end, interval=interval, group_by='ticker')
    adj_close = data.iloc[:, data.columns.get_level_values(1)=='Adj Close'].droplevel(1, axis = 1).add_suffix("_adj_close")
    volume = adj_close = data.iloc[:, data.columns.get_level_values(1)=='Volume'].droplevel(1, axis = 1).add_suffix("_volume")

    extra_info =  pd.DataFrame()
    for stock in stock_tickers:
        ticker = yf.Ticker(stock)
        history = ticker.history(start=start, end=end, interval=interval)
        extra_info[f"{stock}_dividends"] = history['Dividends']
        extra_info[f"{stock}_stock_plits"] = history['Stock Splits']
        extra_info[f"{stock}_sector"] = ticker.info['sector']

    return adj_close, volume, extra_info

def get_returns(data, current_offset = 0, comparison_offset = 1):
    return (data.shift(current_offset) -  data.shift(comparison_offset)) / data.shift(comparison_offset)

def create_dataset(stock_tickers, start, end, interval):
    adj_close, volume, extra_info = get_data(stock_tickers, start, end, interval)
    returns = pd.DataFrame()
    lags = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [0, 2], [0, 5], [0, 10], [0, 15], [0, 30]]
    for lag in lags:
        for col in adj_close.columns:
            returns[f"{col.split()[0]}_r_{lag[0]}_{lag[1]}"] = get_returns(adj_close.loc[:, col], lag[0], lag[1])
 
    sector_map = {sector:i for (i,sector) in enumerate(extra_info[extra_info.select_dtypes(['object']).columns].iloc[0,:].unique())}
    extra_info = extra_info.replace(sector_map)
    return pd.concat([returns, volume / 1000000, extra_info], axis = 1).dropna().astype("float32")


stock_index = '^DJI'
start = "2008-02-19"
end = "2018-02-19"
interval = "1d"
stock_tickers = ['MMM', 'AXP']
features = ['Close','High']  


df = create_dataset(stock_tickers, start, end, interval)
print(df)

#sector, exchange, bookValue, priceToBook, sharesShort


#msft = yf.Ticker("MSFT")

# get stock info
#print(msft.info)

# get historical market data
#hist = msft.history(period="max")

# show actions (dividends, splits)
#print(hist.info)
