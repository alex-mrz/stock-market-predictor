import mplfinance as mpf
import numpy as np
import matplotlib.pyplot as plt

def plot(test, preds_series):

    df_plot = test.copy()

    # Make sure the predictions index aligns with df_plot's index.

    # df_plot = df_plot.join(predictions["Predictions"], how="left")  
    # ^ to plot in backtesting with df with target column. if use, change param to predictions

    df_plot['Predictions'] = preds_series

    # Add markers for Buy and Sell signals
    df_plot["Buy_Marker"] = np.nan
    df_plot["Sell_Marker"] = np.nan

    # Buy signals
    buy_signals = df_plot[df_plot["Predictions"] == 2]
    df_plot.loc[buy_signals.index, "Buy_Marker"] = buy_signals["Low"] * 0.99

    # Sell signals
    sell_signals = df_plot[df_plot["Predictions"] == 0]
    df_plot.loc[sell_signals.index, "Sell_Marker"] = sell_signals["High"] * 1.01


    # apds = [
    #     mpf.make_addplot(df_plot["Buy_Marker"], type='scatter', markersize=100, marker='^', color='green'),
    #     mpf.make_addplot(df_plot["Sell_Marker"], type='scatter', markersize=100, marker='v', color='red')
    # ]
    apds = []
    if not df_plot["Buy_Marker"].dropna().empty:
        apds.append(
            mpf.make_addplot(df_plot["Buy_Marker"], type='scatter', markersize=100, marker='^', color='green')
        )
    if not df_plot["Sell_Marker"].dropna().empty:
        apds.append(
            mpf.make_addplot(df_plot["Sell_Marker"], type='scatter', markersize=100, marker='v', color='red')
        )

    #doesnt work:
    # fig, axlist = mpf.plot(df_plot, type='candle', volume=True, addplot=apds,
    #                  title='SP500 Candlestick Chart with Buy/Sell Signals',
    #                  style='yahoo', returnfig=True)
    
    # last_date = df_plot.index[-1]
    # last_close = df_plot["Close"].iloc[-1]

    # ax = axlist[0]

    # ax.annotate(last_date.strftime('%Y-%m-%d'),
    #             xy=(last_date, last_close),
    #             xycoords='data',
    #             xytext=(0.9, 0.9),
    #             arrowprops=dict(facecolor='black', arrowstyle='->'),
    #             fontsize=10,
    #             color = "blue")
    
    # fig.show()
    # plt.show()

    mpf.plot(df_plot, type='candle', volume=True, addplot=apds,
            title='SP500 Candlestick Chart with Buy/Sell Signals', style='yahoo')