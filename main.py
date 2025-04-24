
from sklearn.metrics import classification_report, precision_score
from data_processing import prepare_data
from modeling import predict, backtest, get_model
from trade_simulator import TradingSim
from plotting import plot
import numpy as np
import pandas as pd



def main(ticker):

    data = pd.DataFrame(prepare_data(ticker))
    
    
    # train = data.iloc[:-200] only for additional testing
    test = data.iloc[-250:]

    predictors = ["Close", "Volume", "Open", "High", "Low", "RSI", "EMA_12", "EMA_26", "MACD", 
                "MACD_Signal", "MACD_Diff", "BB_Mid", "BB_High", "BB_Low", "ATR", "ADX", "OBV",
                  "BB_Width", "ROC"]
    
    model = get_model()

    # new_pred = horizon(data)
    
    # for i in range (len(new_pred)):
    #     predictors.append(new_pred[i])


    predictions, average_profit, profits = backtest(data, model, predictors, ticker)
    # returns = []
    # for i in range (len(profits)):
    #     returns.append(profits[i]/10000 -1)     #only for test by chunks
    
    # returns = np.array(returns)
    # average_return = np.mean(returns)
    # volatility = np.std(returns)
 
    # risk_free_rate = 0
    # sharpe_ratio = (average_return - risk_free_rate) / volatility


    # print(classification_report(predictions["Target"], predictions["Predictions"])) #for backtesting
    # print(f"Profits: {profits}")
    print("sell: 0, Buy: 2, Hold: 1")
    # print(f"Average profit: {average_profit}")
    # print(f"Sharpe Ratio: {sharpe_ratio}")
    plot(test, predictions)
    

# tickers = ["TSLA", "NFLX", "REGN"]

# for i in range(len(tickers)):
#     main(tickers[i])
#     print("\n\n")

main("TSLA")