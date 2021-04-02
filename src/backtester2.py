from datetime import datetime
from dataloader import _DataLoader
from analyzer import Analyzer
from fab_strategy import FabStrategy
from trading_history import TradeHistory
from trade import Trade
from helper import Helper
import pandas as pd
import os


class Backtester2:
    """
    Purpose is to test strategy in history to see its performance

    Attributes
    -----------
    trade_history: TradeHistory Object - repr: list of lists Ex.  [['Long', 'Enter', '2018-04-12 12:46:00', 7696.85, 'Rule 3'],
                                                                    ['Long', 'Exit', '2018-04-13 08:05:00', 8125.01, 'Rule 1']]

    csvUrl:      str             - url of the data
    tf:          int             - timeframe that you want to backtest in.
    start:       str             - starting date
    end:         str             - end_date
    summary:     str             - Analyzed metrics saved in a statement
    loader:      DataLoader Obj  - used to load data.
    symbol_data: pd.DataFrame    - All 1 minute data loaded from csv of given asset
    range_data:  pd.DataFrame    - Sliced 1 minute data from the loaded asset between two dates
    tf_data:     pd.DataFrame    - Abstracted data of "range_data", grouped by tf.


    Methods
    ----------
    set_asset
    set_date_range
    set_timeframe
    validate_trades
    start_backtest


    Please look at each method for descriptions
    """

    def __init__(self):
        self.loader = _DataLoader()
        self.date_range = None
        self.symbol_data = None
        self.start = None
        self.end = datetime.today().strftime('%Y-%m-%d')
        self.tf = None
        self.trade_history = TradeHistory()
        self.tf_data = None
        self.pnl = None
        self.trades = None
        self.summary = "Nothing to Show Yet"

    def value_exists_in_database(self, column, value, table_name = "candlesticks") -> bool:
        cursor = self.loader.conn.cursor()
        df = self.loader.sql.SELECT(f"CAST(CASE WHEN COUNT(DISTINCT({column}))> 0 THEN 1 ELSE 0 END AS BIT) from "
                                    f"{table_name} where {column} = '{value}' ;", cursor)
        return bool(int(df.iloc[0, 0]))

    def set_symbol(self, symbol: str) -> str:
        """
        Parameters
        ----------
        symbol: ticker symbol (e.g. BTCUSDT, ETHUSDT, TSLA)

        :return timeframe
        """
        if not self.value_exists_in_database("symbol", symbol):
            raise Exception("Symbol requested is not in the database yet")

        self.symbol = symbol
        return self.symbol

    def set_timeframe(self, tf: int) -> int:
        """
        Parameters
        ----------
        tf: timeframe

        :return timeframe
        """

        if not self.value_exists_in_database("timeframe", tf):
            raise Exception("Timeframe requested is not in the database yet")

        self.tf = tf
        return self.tf

    def set_date_range(self, start_date: str, end_date: str = None) -> (str, str):
        """
        Paramters:
        ------------
        start: str - start date (in format "YYYY-MM-DD")
        end:   str - end date   (in format "YYYY-MM-DD")

        :return (str, str)
        """
        self.start_date = start_date
        self.end_date = end_date
        return self.start_date, self.end_date

    def load_backtesting_data(self, symbol=None, tf=None, start_date=None, end_date=None, table_name="candlesticks"):
        cursor = self.loader.conn.cursor()
        symbol = self.symbol if not symbol else symbol
        tf = self.tf if not tf else tf
        start_date = self.start_date if not start_date else start_date
        end_date = self.end_date if not end_date else end_date

        self.df = self.loader.sql.SELECT(f"* FROM {table_name} WHERE SYMBOL = '{symbol}' and TIMEFRAME = {tf} AND DATE "
                                    f"BETWEEN '{start_date}' AND '{end_date}' ORDER BY timestamp", cursor)

        return self.df

    def validate_trades(self, trade_index: [float], trade_history: TradeHistory) -> [[[]]]:
        """ Puts the already given information in a way that is easily debuggable
            Groups TradeHistory object by "Enter" & "Exit" and adds the corresponding trade index to it.[

        trade_index: a list of pnl percentages
        trade_history: TradeHistory object which is just a list of Trade objects.
         Example:
             [['Long', 'Enter', '2017-10-05 15:43:00', 4333.59, 'Rule 1'],
             ['Long', 'Exit', '2017-10-17 03:38:00', 5669.68, 'Rule 1'],
             [30.7264]]
        """
        # Ex. [1.25,1.35,0.91,1.04] -> [25,35,-9,4]
        trade_index_percentage = Helper.factor_to_percentage(trade_index)
        trade_history_range = range(1, len(self.trade_history), 2)
        trade_index_range = range(len(trade_index_percentage))

        compiled_list = []

        for i, j in zip(trade_history_range, trade_index_range):
            compiled_list.append(self.trade_history[i:i + 2] + [[trade_index_percentage[j]]])

        return compiled_list

    def check_rule_1(self, strategy, i, list_of_str_dates):
        if strategy.rule_1_buy_enter(i) and self.trade_history.last_trade().status != "Enter":
            self.trade_history.append(Trade(["Long", "Enter", list_of_str_dates[i], strategy.price[i], "Rule 1"]))

        elif strategy.rule_1_buy_exit(i) and self.trade_history.last_trade().side == "Long" \
                and self.trade_history.last_trade().status == "Enter":
            self.trade_history.append(Trade(["Long", "Exit", list_of_str_dates[i], strategy.price[i], "Rule 1"]))

        elif strategy.rule_1_short_enter(i) and self.trade_history.last_trade().status != "Enter":
            self.trade_history.append(Trade(["Short", "Enter", list_of_str_dates[i], strategy.price[i], "Rule 1"]))

        elif strategy.rule_1_short_exit(i) and self.trade_history.last_trade().side == "Short" \
                and self.trade_history.last_trade().status == "Enter":
            self.trade_history.append(Trade(["Short", "Exit", list_of_str_dates[i], strategy.price[i], "Rule 1"]))

    def check_rule_2(self, strategy, i, list_of_str_dates, sensitivity):
        if strategy.rule_2_buy_enter(i, sensitivity) and self.trade_history.last_trade().status != "Enter":
            self.trade_history.append(Trade(["Long", "Enter", list_of_str_dates[i], strategy.black[i], "Rule 2"]))

        elif strategy.rule_2_buy_stop(i) and self.trade_history.last_trade().rule == "Rule 2" and \
                self.trade_history.last_trade().side == "Long" and self.trade_history.last_trade().status == "Enter":
            self.trade_history.append(Trade(["Long", "Exit", list_of_str_dates[i], strategy.price[i], "Rule 2"]))

        elif strategy.rule_2_short_enter(i, sensitivity) and self.trade_history.last_trade().status != "Enter":
            self.trade_history.append(Trade(["Short", "Enter", list_of_str_dates[i], strategy.black[i], "Rule 2"]))

        elif strategy.rule_2_short_stop(i) and self.trade_history.last_trade().rule == "Rule 2" and \
                self.trade_history.last_trade().side == "Short" and self.trade_history.last_trade().status == "Enter":
            self.trade_history.append(Trade(["Short", "Exit", list_of_str_dates[i], strategy.price[i], "Rule 2"]))

    def check_rule_3(self, strategy, i, list_of_str_dates):
        if strategy.rule_3_buy_enter(i) and self.trade_history.last_trade().status != "Enter":
            self.trade_history.append(Trade(["Long", "Enter", list_of_str_dates[i], strategy.price[i], "Rule 3"]))

        elif strategy.rule_3_short_enter(i) and self.trade_history.last_trade().status != "Enter":
            self.trade_history.append(Trade(["Short", "Enter", list_of_str_dates[i], strategy.price[i], "Rule 3"]))

    def start_backtest(self, df: pd.DataFrame, strategy: FabStrategy, sensitivity: float) -> str:
        """
        Tests the asset in history, with respect to the rules outlined in the FabStrategy class.
        It adds applicable trades to a list and then an Analyzer object summarizes the profitability

        Parameters:
        -----------
        df:          pd.DataFrame - The abstracted tf data that is ready for backtesting. This is not minute data if tf != 1
        strategy:    Object - any trading strategy that takes the index and sensitivity as input, and returns boolean values.
        sensitivity: float  - Allowance between price and MA. The larger the value, the further and less sensitive.


        :return str - A summary of all metrics in the backtest. (See Analyzer.summarize_statistics method for more info)
        """

        self.trade_history = TradeHistory()

        list_of_str_dates = Helper.timestamp_object_to_string(df['date'])

        # Creating necessary moving averages from FabStrategy class
        strategy.load_data(df)
        strategy.update_moving_averages()

        # Iterating through every single data point and checking if rules apply.
        for row_index in range(231, len(df) - 1):
            self.check_rule_1(strategy, row_index, list_of_str_dates)
            self.check_rule_2(strategy, row_index, list_of_str_dates, sensitivity)
            self.check_rule_3(strategy, row_index, list_of_str_dates)

        # Analyzing the trade history
        analyze_backtest = Analyzer()
        analyze_backtest.calculate_statistics(self.trade_history)

        # Adding all trades in a list. They are in the form of: 1+profit margin. Ex. [1.04, 0.97, 1.12] etc.
        self.trades = analyze_backtest.get_trades()
        self.pnl = round(analyze_backtest.get_pnl(self.trades), 3)
        self.summary = analyze_backtest.summarize_statistics()

        return self.summary


