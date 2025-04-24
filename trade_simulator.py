import numpy as np

class TradingSim:
    def __init__(self, init_capital=1000):
        self.capital = init_capital
        self.trade_log = [] # a fucking log that pisses me off
        self.active_trades = {}
        self.key_ID = 0
        self.investment = 25


    def log_trade(self, trade_type, entry_price, exit_price, investment, profit_loss, tp, sl):

        trade_info = {
            "Trade Type": trade_type,
            "Entry Price": entry_price,
            "Exit Price": exit_price,
            "Investment": investment,
            "Profit/Loss": profit_loss,
            "TP": tp,
            "SL": sl,
        }
        self.trade_log.append(trade_info)

    
    @staticmethod
    def sl_tp_buy(risk, atr, stock_price):
        sl = stock_price - (1.5*atr)
        tp = stock_price + (risk*1.5*atr)
        return tp, sl
    @staticmethod
    def sl_tp_sell(risk, atr, stock_price):
        sl = stock_price + (1.5*atr)
        tp = stock_price - (risk*1.5*atr)
        return tp, sl

    def execute_trade(self, trade_type, entry_price, risk, atr):
        if trade_type == "Buy":
            tp, sl = self.sl_tp_buy(risk, atr, entry_price)
            # self.capital -= self.investment #pretty sure useless
            self.active_trades[self.key_ID] = {
                "Trade Type": trade_type,
                "Entry Price": entry_price,
                "Investment": self.investment,
                "TP": tp,
                "SL": sl
            }
            self.key_ID += 1
        elif trade_type == "Sell":
            tp, sl = self.sl_tp_sell(risk, atr, entry_price)
            # self.capital -= self.investment #pretty sure useless
            self.active_trades[self.key_ID] = {
                "Trade Type": trade_type,
                "Entry Price": entry_price,
                "Investment": self.investment,
                "TP": tp,
                "SL": sl
            }
            self.key_ID += 1
        else:
            return

    def close_trade(self, current_price, leverage=2):
        for key in list(self.active_trades.keys()):
            if self.active_trades[key]["Trade Type"] == "Buy":
                if current_price >= self.active_trades[key]["TP"] or current_price <= self.active_trades[key]["SL"]:
                    profit = (current_price/self.active_trades[key]["Entry Price"]) * self.investment 
                    true_profit = (profit - self.investment)  *leverage 
                    self.capital += true_profit
                    # print(profit) #debug
                    # print(self.capital) #debug
                    self.log_trade("Buy", self.active_trades[key]["Entry Price"], current_price, 
                                   self.investment, true_profit, self.active_trades[key]["TP"], self.active_trades[key]["SL"])
                    del self.active_trades[key]
                
                
            elif self.active_trades[key]["Trade Type"] == "Sell":
                if current_price <= self.active_trades[key]["TP"] or current_price >= self.active_trades[key]["SL"]:
                    profit = (self.active_trades[key]["Entry Price"]/current_price) * self.investment 
                    true_profit = (profit - self.investment) * leverage 
                    self.capital += true_profit
                    # print(profit) #debug
                    # print(self.capital) #debug
                    self.log_trade("Sell",
                                    self.active_trades[key]["Entry Price"],current_price, self.investment,
                                      true_profit, self.active_trades[key]["TP"], self.active_trades[key]["SL"])
                    del self.active_trades[key]

    def finish_trades(self, current_price, leverage=2):
        for key in list(self.active_trades.keys()):
            if self.active_trades[key]["Trade Type"] == "Buy":
                profit = (current_price/self.active_trades[key]["Entry Price"]) * self.investment 
                true_profit = (profit - self.investment)* leverage
                self.capital += true_profit
                # print(profit) #debug
                # print(self.capital) #debug
                self.log_trade("Buy", self.active_trades[key]["Entry Price"], current_price, 
                               self.investment, true_profit, self.active_trades[key]["TP"], self.active_trades[key]["SL"])
                del self.active_trades[key]
                
            elif self.active_trades[key]["Trade Type"] == "Sell":
                profit = (self.active_trades[key]["Entry Price"]/current_price) * self.investment 
                true_profit = (profit - self.investment) * leverage
                self.capital += true_profit
                # print(profit) #debug
                # print(self.capital) #debug
                self.log_trade("Sell",
                                self.active_trades[key]["Entry Price"],current_price, self.investment,
                                  true_profit, self.active_trades[key]["TP"], self.active_trades[key]["SL"])
                del self.active_trades[key]

   

        