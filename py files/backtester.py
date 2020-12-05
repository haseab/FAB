from datetime import datetime
from dataloader import DataLoader
from analyzer import Analyzer
from strategy_executor import StrategyExecutor

class Backtester():
    def __init__(self):
        self.trade_history = []
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
        return (self.start, self.end)

    def set_timeframe(self, tf):
        self.tf = tf
        self.tf_data = self.loader.timeframe_setter(self.minute_data, self.tf)
        return self.tf

    def start_backtest(self):
        executor = StrategyExecutor()
        analyze_backtest = Analyzer()

        self.trade_history = executor.fab_strategy(self.tf_data)
        analyze_backtest.calculate_statistics(self.trade_history)
        self.summary = analyze_backtest.summarize_statistics()
        return self.summary
