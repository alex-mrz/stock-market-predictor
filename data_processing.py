import yfinance as yf
import pandas as pd
import ta
import json
from ibkr_data import ibkrdatafetcher

# fetcher = ibkrdatafetcher()
#beneath to change later on because rn just using this func for start date and train days
def specific(name="Netflix"):
    with open("stocks.json", "r") as f:
        stocks = json.load(f)

    stock_info = stocks[name]
    ticker = stock_info["ticker"]
    start_date = stock_info["start_date"]
    train_days = stock_info["train_days"]
    return ticker, start_date, train_days

ticker1, start_date1, train_days = specific()


def load_data(ticker, period="max"):
    data = yf.Ticker(ticker).history(period=period)
    for col in ["Dividends", "Stock Splits"]:
        if col in data.columns:
            del data[col]
    # data.index = data.index.tz_localize(None)
    # last_date = data.index[-1].strftime("%Y-%m-%d")
    # period = '1 M'
    # data1 = fetcher.fetch_data(ticker, period)
    # data1.index = data1.index.tz_localize(None)

    # data1 = data1.loc[data1.index > last_date]

    # combined = pd.concat([data, data1])
    # fetcher.disconnect()


    return data

# data = load_data()
# print(data)
# print(load_data()) 

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
    
    def add_rsi(self, window=14):
        self.df["RSI"] = ta.momentum.RSIIndicator(close=self.df["Close"].shift(1), window=window).rsi()
    
    def add_ema(self, short_window=12, long_window=26):
        self.df["EMA_12"] = ta.trend.EMAIndicator(close=self.df["Close"].shift(1), window=short_window).ema_indicator()
        self.df["EMA_26"] = ta.trend.EMAIndicator(close=self.df["Close"].shift(1), window=long_window).ema_indicator()
    
    def add_macd(self):
        macd = ta.trend.MACD(close=self.df["Close"].shift(1))
        self.df["MACD"] = macd.macd()
        self.df["MACD_Signal"] = macd.macd_signal()
        self.df["MACD_Diff"] = macd.macd_diff()
    
    def add_bollinger_bands(self, window=20, window_dev=2):
        bb = ta.volatility.BollingerBands(close=self.df["Close"].shift(1), window=window, window_dev=window_dev)
        self.df["BB_Mid"] = bb.bollinger_mavg()
        self.df["BB_High"] = bb.bollinger_hband()
        self.df["BB_Low"] = bb.bollinger_lband()
    
    def add_atr(self, window=14):
        atr = ta.volatility.AverageTrueRange(high=self.df["High"].shift(1), low=self.df["Low"].shift(1), close=self.df["Close"].shift(1), window=window)
        self.df["ATR"] = atr.average_true_range()
    
    def add_adx(self, window=14):
        adx = ta.trend.ADXIndicator(high=self.df["High"].shift(1), low=self.df["Low"].shift(1), close=self.df["Close"].shift(1), window=window)
        self.df["ADX"] = adx.adx()
    
    def add_obv(self):
        obv = ta.volume.OnBalanceVolumeIndicator(close=self.df["Close"].shift(1), volume=self.df["Volume"].shift(1))
        self.df["OBV"] = obv.on_balance_volume()
    
    def add_bb_width(self):
        self.df["BB_Width"] = self.df["BB_High"] - self.df["BB_Low"]
    
    def add_roc(self, window=14):
        self.df["ROC"] = self.df["Close"].shift(1).pct_change(periods=window) * 100
    
    def process(self):
        self.add_rsi()
        self.add_ema()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_atr()
        self.add_adx()
        self.add_obv()
        self.add_bb_width()
        self.add_roc()
        return self.df

def prepare_data(ticker, start_date=start_date1):
    data = load_data(ticker)
    # print(data)
    fe = FeatureEngineer(data)
    data_processed = fe.process()


    data_processed["Weekly"] = data_processed["Close"].shift(-5)
    data_processed["Target"] = 0

    data_processed.loc[data_processed["Weekly"] > data_processed["Close"] * 1.02, "Target"] = 1  # Buy
    data_processed.loc[data_processed["Weekly"] < data_processed["Close"] * 0.98, "Target"] = -1  # Sell

    data_processed["Target"] = data_processed["Target"].map({-1: 0, 0: 1, 1: 2})

    # # start_date = pd.to_datetime(start_date)
    # # data_processed = data_processed.loc[data_processed.index >= start_date].copy()
    # # data_processed = data_processed.dropna()

    
    # start_date = pd.to_datetime(start_date)  # This will be tz-naive by default
    # data_processed = data_processed.loc[data_processed.index >= start_date].copy()

    data_processed = data_processed.loc[start_date:].copy()
    # data_processed = data_processed.dropna()
    return data_processed


# data = pd.DataFrame(prepare_data())
# print(data)



