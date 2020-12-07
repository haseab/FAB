from datetime import datetime
from dataloader import DataLoader
from analyzer import Analyzer
from strategy_executor import StrategyExecutor


class Backtester():
    def __init__(self):
        self.trade_history = ['List of Trades']
        self.csvUrl = r"BTCUSDT Aug 17 2017 to Dec 5 2020.csv"
        self.tf = None
        self.start = None
        self.end = datetime.today().strftime('%Y-%m-%d')
        self.summary = "Nothing to Show Yet"
        self.loader = DataLoader()

    def set_asset(self, symbol, csvUrl):
        """Asset to be backtesting"""
        # Set CSV Url or connect to DB
        self.symbol_data = self.loader.load_csv(csvUrl)
        return self.symbol_data

    def set_date_range(self, start, end=None):
        self.start = start
        if end != None:
            self.end = end
        self.minute_data = self.loader.get_range(self.symbol_data, self.start, self.end)
        self.tf_data = self.loader.timeframe_setter(self.minute_data, self.tf)
        return (self.start, self.end)

    def set_timeframe(self, tf):
        self.tf = tf
        return self.tf

    def backtest(self, strategy):
        df = self.tf_data
        self.trade_history = ['List of Trades']
        strategy.load_data(df)
        strategy.create_objects()
        date = [str(df['Datetime'].iloc[i]) for i in range(len(df['Datetime']))]

        for i in range(231, len(df) - 1):
            if strategy.rule_1_buy_enter(i) and self.trade_history[-1][1] != "Enter":
                self.trade_history.append(["Long", "Enter", date[i + 1], strategy.price[i + 1], "Rule 1"])
            elif strategy.rule_1_buy_exit(i) and self.trade_history[-1][:2] == ["Long", 'Enter']:
                self.trade_history.append(["Long", "Exit", date[i + 1], strategy.price[i + 1], "Rule 1"])
            elif strategy.rule_1_short_enter(i) and self.trade_history[-1][1] != "Enter":
                self.trade_history.append(["Short", "Enter", date[i + 1], strategy.price[i + 1], "Rule 1"])
            elif strategy.rule_1_short_exit(i) and self.trade_history[-1][:2] == ["Short", 'Enter']:
                self.trade_history.append(["Short", "Exit", date[i + 1], strategy.price[i + 1], "Rule 1"])
            elif strategy.rule_2_buy_enter(i) and self.trade_history[-1][1] != "Enter":
                self.trade_history.append(["Long", "Enter", date[i + 1], strategy.black[i + 1], "Rule 2"])
            elif strategy.rule_2_buy_stop(i) and self.trade_history[-1][-1] == "Rule 2" and self.trade_history[-1][
                                                                                            :2] == ["Long", 'Enter']:
                self.trade_history.append(["Long", "Exit", date[i + 1], strategy.price[i + 1], "Rule 2"])
            elif strategy.rule_2_short_enter(i) and self.trade_history[-1][1] != "Enter":
                self.trade_history.append(["Short", "Enter", date[i + 1], strategy.black[i + 1], "Rule 2"])
            elif strategy.rule_2_short_stop(i) and self.trade_history[-1][:2] == ["Short", 'Enter'] and \
                    self.trade_history[-1][-1] == "Rule 2":
                self.trade_history.append(["Short", "Exit", date[i + 1], strategy.price[i + 1], "Rule 2"])
            elif strategy.rule_3_buy_enter(i) and self.trade_history[-1][1] != "Enter":
                self.trade_history.append(["Long", "Enter", date[i + 1], strategy.price[i + 1], "Rule 3"])
            elif strategy.rule_3_short_enter(i) and self.trade_history[-1][1] != "Enter":
                self.trade_history.append(["Short", "Enter", date[i + 1], strategy.price[i + 1], "Rule 3"])

        analyze_backtest = Analyzer()
        analyze_backtest.calculate_statistics(self.trade_history)
        self.trades = analyze_backtest.trades
        self.profit = round(analyze_backtest.capital / analyze_backtest.initial_capital, 3)
        self.summary = analyze_backtest.summarize_statistics()
        return self.summary