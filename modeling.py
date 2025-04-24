import pandas as pd
from trade_simulator import TradingSim
from sklearn.ensemble import RandomForestClassifier
from plotting import plot
from xgboost import XGBClassifier
import numpy as np
from data_processing import specific

ticker1, start_date1, train_days = specific()




# def get_model():
#     model = RandomForestClassifier(n_estimators=450, min_samples_split=50,
#                                 random_state=1, class_weight={1:50, -1:35, 0:15}, n_jobs=3)
#     return model

def get_model():
    model = XGBClassifier(
        n_estimators=850,
        max_depth=2,           
        learning_rate=0.004,
        min_child_weight=12,   
        gamma=10,  
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.6,
        reg_lambda=5,
        random_state=1,
        n_jobs = 3
    )
    return model

def dynamic_t(atr_value, current_price, k=1.05, base_threshold_buy=0.65, base_threshold_sell = 0.66):

    buy_t = base_threshold_buy - ((atr_value/current_price)*k)
    sell_t = base_threshold_sell - ((atr_value/current_price)*k)

    if atr_value/current_price > 0.25:
        buy_t = 0.75
        sell_t = 0.75
    else:
        buy_t = max(buy_t, 0.56)
        sell_t = max(sell_t, 0.57)

    return buy_t, sell_t


def predict(train, test, predictors, model):

    unique, counts = np.unique(train["Target"], return_counts=True)
    total = len(train["Target"])
    class_weights = {cls: total / count for cls, count in zip(unique, counts)}
    base_weights = {
    0: class_weights[0] * 1.9,  #1.7  1.8 1.9 1.8
    1: class_weights[1] *0.1,      #0.4  .2 .2 .1
    2: class_weights[2] * 2.5       #2.1  2.3 2.5 2.3
}
    sample_weight = np.array([base_weights[y] for y in train["Target"]])           #only for xgb

    model.fit(train[predictors], train["Target"], sample_weight=sample_weight)

    # print(test.columns)
    test = test.drop(columns=["Target"])
    test = test.drop(columns=["Weekly"])

    proba = model.predict_proba(test[predictors])
    
    # Get the mapping of classes (e.g., [-1, 0, 1] for rfc, [0,1,2] for xgb)
    class_map = list(model.classes_)
    buy_idx = class_map.index(2) if 2 in class_map else None #1 if rfc
    sell_idx = class_map.index(0) if 0 in class_map else None #-1 if rfc

    
    
    preds = []
    simulator = TradingSim()
    a = len(test) - 1
    for i in range(len(test)):
        # Get confidence for Buy and Sell using the proper indices
        buy_conf = proba[i][buy_idx] if buy_idx is not None else 0
        sell_conf = proba[i][sell_idx] if sell_idx is not None else 0

        if i >= 15: 
            atr_average = test["ATR"].shift(1).rolling(14).mean().iloc[i] 
            price_current = test["Open"].iloc[i]
            buy_t, sell_t = dynamic_t(atr_average, price_current)
        elif i>=1:
            atr_average = test["ATR"].shift(1).rolling(i).mean().iloc[i]
            price_current = test["Open"].iloc[i]
            buy_t = 0.65
            sell_t = 0.66
        else:
            atr_average = test["ATR"].iloc[i]
            price_current = test["Open"].iloc[i]
            buy_t = 0.65
            sell_t = 0.66
        
        if buy_conf > buy_t:
            preds.append(2) #1 if rfc
            simulator.execute_trade("Buy", test["Open"].iloc[i], 2, atr_average) #number is risk factor fo tp/sl

        elif sell_conf > sell_t :
            preds.append(0) #-1 if rfc
            simulator.execute_trade("Sell", test["Open"].iloc[i], 2, atr_average)

        else:
            preds.append(1) #0 if rfc

        simulator.close_trade(test["Open"].iloc[i])
    simulator.finish_trades(test["Open"].iloc[a])
    # print(test)
    

    
    
    preds_series = pd.Series(preds, index=test.index, name="Predictions")
    # combined = pd.concat([test["Target"], preds_series], axis=1) #for backtesting and comparing preds with target

    
    return preds_series, simulator.capital


def backtest(data, model, predictors, ticker, start=train_days, step=250):
    all_predictions = []
    profits = []
    average_profit = 0
    total = 0

    train = data.iloc[:-250].copy() #test on whole data
    test = data.iloc[-250:]

    predictions, tot = predict(train, test, predictors, model)
    all_predictions.append(predictions)
    profits.append(tot)
    print(predictions[-11:]) #debug
    print(predictions[len(predictions)-1:]) #last prediction
    print("\n")
    a = predictions.iloc[-1] #last prediction
    if a == 2:
        tp, sl = TradingSim.sl_tp_buy(2, test["ATR"].iloc[-1], test["Open"].iloc[-1])
        tp_dollar = (tp/test["Open"].iloc[-1] * 25 -25) *2 #2 is leverage
        sl_dollar = (sl/test["Open"].iloc[-1] * 25 - 25) *2
        print(f"BUY {ticker} stock at {test['Open'].iloc[-1]}, with TP: ${tp_dollar} and SL: ${sl_dollar}\n")
    elif a == 0:
        tp, sl = TradingSim.sl_tp_sell(2, test["ATR"].iloc[-1], test["Open"].iloc[-1])
        tp_dollar = (test["Open"].iloc[-1]/tp * 25 - 25) * 2
        sl_dollar = (test["Open"].iloc[-1]/sl * 25 - 25) * 2
        print(f"SELL {ticker} stock at {test['Open'].iloc[-1]}, with TP: ${tp_dollar} and SL: ${sl_dollar}\n")
    else:
        print(f"HOLD {ticker} stock at {test['Open'].iloc[-1]}\n")

    print("\n\n")
    print("Last date in backtest:", test.index[-1])
    print(tot)

    
    # for i in range(start, data.shape[0], step):  #only for test by chunks
    #     train = data.iloc[0:i].copy()
    #     test = data.iloc[i:(i+step)].copy()
    #     predictions, tot = predict(train, test, predictors, model)
    #     all_predictions.append(predictions)
    #     # if tot > 14000:
    #     #     print(tot)                #debug!!
    #     #     plot(test, predictions)
    #     profits.append(tot)
    #     print(tot) #test profit by chunks
    # for i in range(len(profits)):
    #     total += profits[i]
    # average_profit = total / len(profits)
    # last_date = test.index[-1]
    # print("Last date in backtest:", last_date)
    return pd.concat(all_predictions), average_profit, profits
    

def calculate_trend_row(x, threshold=0.02):
    """
    x: A pandas Series of closing prices of length `horizon`
    threshold: The cutoff percentage (in decimal form) for a meaningful change.
               For example, 0.02 means 2%.
    
    Returns:
       2 if the relative change > threshold,
       0 if the relative change < -threshold,
       1 otherwise.
    """
    # Ensure the window is not empty or contains NaN
    if len(x) == 0 or x.iloc[0] == 0:
        return 1  # Default to neutral if we cannot compute change properly

    # Calculate relative change: (last / first) - 1
    rel_change = (x.iloc[-1] / x.iloc[0]) - 1
    
    if rel_change > threshold:
        return 2
    elif rel_change < -threshold:
        return 0
    else:
        return 1




# def horizon(data_processed):
#     horizons = [2, 5, 14, 30]
#     new_predictors = []
#     for horizon in horizons:
#         # Price-based features :
#         # Compute rolling average 
#         rolling_avg_close = data_processed["Close"].shift(1).rolling(horizon).mean()
        
#         # Create a price ratio: current Close / rolling average Close
#         price_ratio_col = f"Close_Ratio_{horizon}"
#         data_processed[price_ratio_col] = data_processed["Close"].shift(1) / rolling_avg_close
        
#         # Create a price trend feature based on your Target (shifted to avoid lookahead bias)
#         price_trend_col = f"Trend_{horizon}"
#     #     data_processed[price_trend_col] = data_processed["Close"].shift(1).rolling(horizon).apply(
#     #     lambda x: calculate_trend_row(x, threshold=0.02)
#     # )
#         data_processed[price_trend_col] = data_processed.shift(1).rolling(horizon).sum()["Target"]
#         # Add these new feature names to your predictors list
#         new_predictors += [price_ratio_col, price_trend_col]
        
#         #Volume-based features :
#         # Compute rolling average 
#         rolling_avg_volume = data_processed["Volume"].shift(1).rolling(horizon).mean()
        
#         # Create a volume ratio: current Volume / rolling average Volume
#         volume_ratio_col = f"Volume_Ratio_{horizon}"
#         data_processed[volume_ratio_col] = data_processed["Volume"].shift(1) / rolling_avg_volume
        
#         # Create a volume trend: difference between current Volume and its rolling average
#         volume_trend_col = f"Volume_Trend_{horizon}"
#         data_processed[volume_trend_col] = data_processed["Volume"].shift(1) - rolling_avg_volume
        
#         # Add these new feature names to your predictors list
#         new_predictors += [volume_ratio_col, volume_trend_col]


#     # data_processed = data_processed.dropna()
#     return new_predictors

