from datetime import datetime
from dataloader import _DataLoader
from analyzer import Analyzer
from fab_strategy import FabStrategy
import pandas as pd


class Backtester:
    """
    Purpose is to test strategy in history to see its performance

    Attributes
    -----------
    trade_history: TradeHistory Object - repr: list of lists Ex.  [['Long', 'Enter', '2018-04-12 12:46:00', 7696.85, 'Rule 3'],
                                                                    ['Long', 'Exit', '2018-04-13 08:05:00', 8125.01, 'Rule 1']]

    csvUrl:  str             - url of the data
    tf:      int             - timeframe that you want to backtest in.
    start:   str             - starting date
    end:     str             - end_date
    summary: str             - Analyzed metrics saved in a statement
    loader:  DataLoader Obj  - used to load data.

    Methods
    ----------
    set_asset
    set_date_range
    set_timeframe
    start_backtest

    Please look at each method for descriptions
    """

    def __init__(self):
        self.trade_history = TradeHistory()
        self.csvUrl = r"C:\Users\haseab\Desktop\desktop comp\Desktop\Python\PycharmProjects\FAB\data\Binance BTCUSDT Aug 17 2017 to Dec 5 2020.csv"  # Hard Coded
        self.tf = None
        self.start = None
        self.end = datetime.today().strftime('%Y-%m-%d')
        self.summary = "Nothing to Show Yet"
        self.loader = _DataLoader()

    def set_asset(self, symbol: str, csvUrl: str = None) -> pd.DataFrame:
        """Asset to be backtesting"""
        if csvUrl == None:
            csvUrl = self.csvUrl

        # Set CSV Url or connect to DB
        self.symbol_data = self.loader._load_csv(csvUrl)
        return self.symbol_data

    def set_date_range(self, start: str, end: str = None) -> None:
        """
        The data will be sliced such that the respective date range is returned from the dataframe.
        The data is also abstracted by the timframe_setter() method.

        Paramters:
        ------------
        start: str - start date
        end:   str - end date

        :return None
        """
        self.start = start
        if end != None:
            self.end = end
        self.minute_data = self.loader._get_range(self.symbol_data, self.start, self.end)
        self.tf_data = self.loader._timeframe_setter(self.minute_data, self.tf)
        return

    def set_timeframe(self, tf: int) -> int:
        """
        Parameters
        ----------
        tf: timeframe

        :return timeframe
        """
        self.tf = tf
        return self.tf

    def start_backtest(self, strategy: FabStrategy, sensitivity) -> str:
        """
        Tests the asset in history, with respect to the rules outlined in the FabStrategy class.
        It adds applicable trades to a list and then an Analyzer object summarizes the profitability

        Parameters:
        -----------
        strategy:    Object - any trading strategy that takes the index and sensitivity as input, and returns boolean values.
        sensitivity: float  - Allowance between price and MA. The larger the value, the further and less sensitive.


        :return str - A summary of all metrics in the backtest. (See Analyzer.summarize_statistics method for more info)
        """
        df = self.tf_data
        self.trade_history = TradeHistory()

        # Converting Datetime column from Timestamp objects into strings
        date = [str(df['Datetime'].iloc[i]) for i in range(len(df['Datetime']))]

        # Creating necessary moving averages from FabStrategy class
        strategy.load_data(df)
        strategy.create_objects()

        # Iterating through every single data point and checking if rules apply.
        for i in range(231, len(df) - 1):
            # Second condition ensures you aren't doubled entering
            if strategy.rule_1_buy_enter(i) and self.trade_history.last_trade().status != "Enter":
                self.trade_history.append(Trade(["Long", "Enter", date[i], strategy.price[i], "Rule 1"]))

            # Second condition ensures that the previous trade was entering so that it can exit.
            elif strategy.rule_1_buy_exit(i) and self.trade_history.last_trade().side == "Long" \
                    and self.trade_history.last_trade().status == "Enter":
                self.trade_history.append(Trade(["Long", "Exit", date[i], strategy.price[i], "Rule 1"]))

            elif strategy.rule_1_short_enter(i) and self.trade_history.last_trade().status != "Enter":
                self.trade_history.append(Trade(["Short", "Enter", date[i], strategy.price[i], "Rule 1"]))

            elif strategy.rule_1_short_exit(i) and self.trade_history.last_trade().side == "Short" \
                    and self.trade_history.last_trade().status == "Enter":
                self.trade_history.append(Trade(["Short", "Exit", date[i], strategy.price[i], "Rule 1"]))

            elif strategy.rule_2_buy_enter(i, sensitivity) and self.trade_history.last_trade().status != "Enter":
                self.trade_history.append(Trade(["Long", "Enter", date[i], strategy.black[i], "Rule 2"]))

            elif strategy.rule_2_buy_stop(i) and self.trade_history.last_trade().rule == "Rule 2" and \
                    self.trade_history.last_trade().side == "Long" and self.trade_history.last_trade().status == "Enter":
                self.trade_history.append(Trade(["Long", "Exit", date[i], strategy.price[i], "Rule 2"]))

            elif strategy.rule_2_short_enter(i, sensitivity) and self.trade_history.last_trade().status != "Enter":
                self.trade_history.append(Trade(["Short", "Enter", date[i], strategy.black[i], "Rule 2"]))
            elif strategy.rule_2_short_stop(i) and self.trade_history.last_trade().rule == "Rule 2" and \
                    self.trade_history.last_trade().side == "Short" and self.trade_history.last_trade().status == "Enter":
                self.trade_history.append(Trade(["Short", "Exit", date[i], strategy.price[i], "Rule 2"]))

            elif strategy.rule_3_buy_enter(i) and self.trade_history.last_trade().status != "Enter":
                self.trade_history.append(Trade(["Long", "Enter", date[i], strategy.price[i], "Rule 3"]))

            elif strategy.rule_3_short_enter(i) and self.trade_history.last_trade().status != "Enter":
                self.trade_history.append(Trade(["Short", "Enter", date[i], strategy.price[i], "Rule 3"]))

        # Analyzing the trade history
        analyze_backtest = Analyzer()
        analyze_backtest.calculate_statistics(self.trade_history)

        # Adding all trades in a list. They are in the form of: 1+profit margin. Ex. [1.04, 0.97, 1.12] etc.
        self.trades = analyze_backtest.trades

        self.profit = round(analyze_backtest.profit, 3)
        self.summary = analyze_backtest.summarize_statistics()

        return self.summary