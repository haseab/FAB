from datetime import datetime
from dataloader import _DataLoader
from analyzer import Analyzer
from fab_strategy import FabStrategy
from trading_history import TradeHistory
from trade import Trade
from helper import Helper
import pandas as pd
from illustrator import Illustrator
import numpy as np
import os


class Backtester:
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
        self.analyzer = Analyzer()
        self.loader = _DataLoader()
        self.date_range = None
        self.symbol_data = None
        self.start = None
        self.end = datetime.today().strftime('%Y-%m-%d')
        self.tf = None
        self.df = None
        self.df_th = None
        self.df_tf = None
        self.trade_history = TradeHistory()
        self.illustrator = Illustrator()
        self.strategy = FabStrategy()
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
        #
        # if not self.value_exists_in_database("timeframe", tf):
        #     raise Exception("Timeframe requested is not in the database yet")

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

    def load_backtesting_data(self, symbol=None, start_date=None, end_date=None, table_name="candlesticks", all_data=False):
        cursor = self.loader.sql.conn.cursor()
        symbol = self.symbol if not symbol else symbol

        if all_data:
            df = self.loader.sql.SELECT(f"* FROM {table_name} WHERE SYMBOL = '{symbol}' AND TIMEFRAME = '1' ORDER BY timestamp", cursor)

        else:
            start_date = self.start_date if not start_date else start_date
            end_date = self.end_date if not end_date else end_date
            df = self.loader.sql.SELECT(f"* FROM {table_name} WHERE SYMBOL = '{symbol}' AND TIMEFRAME = '1' AND DATE "
                                        f"BETWEEN '{start_date}' AND '{end_date}' ORDER BY timestamp", cursor)
        # df = df.drop('id', axis=1)
        df['date'] = [np.datetime64(date) for date in df['date'].values]
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(
            float)
        self.df = df.set_index('candle_id')
        return self.df

    def load_timeframe_data(self, tf=None):
        tf = self.tf if not tf else tf
        self.df_tf = self.loader._timeframe_setter(self.df, tf)
        return self.df_tf

    def check_rule_1(self, strategy, i, list_of_str_dates):
        if strategy.rule_1_buy_enter(i) and self.trade_history.last_trade().status != "Enter":
            self.trade_history.append(Trade(["Long", "Enter", list_of_str_dates[i+1], strategy.price[i], "Rule 1"]))

        elif strategy.rule_1_buy_exit(i) and self.trade_history.last_trade().side == "Long" \
                and self.trade_history.last_trade().status == "Enter":
            self.trade_history.append(Trade(["Long", "Exit", list_of_str_dates[i+1], strategy.price[i], "Rule 1"]))

        elif strategy.rule_1_short_enter(i) and self.trade_history.last_trade().status != "Enter":
            self.trade_history.append(Trade(["Short", "Enter", list_of_str_dates[i+1], strategy.price[i], "Rule 1"]))

        elif strategy.rule_1_short_exit(i) and self.trade_history.last_trade().side == "Short" \
                and self.trade_history.last_trade().status == "Enter":
            self.trade_history.append(Trade(["Short", "Exit", list_of_str_dates[i+1], strategy.price[i], "Rule 1"]))

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
            self.trade_history.append(Trade(["Long", "Enter", list_of_str_dates[i+1], strategy.price[i], "Rule 3"]))

        elif strategy.rule_3_short_enter(i) and self.trade_history.last_trade().status != "Enter":
            self.trade_history.append(Trade(["Short", "Enter", list_of_str_dates[i+1], strategy.price[i], "Rule 3"]))

    def graph_trade(self, tid=None, index=None, rule=None, large_view=False):
        """Assuming no previous index on trading history"""
        if tid != None:
            print(self.df_th.set_index('trade_id').loc[tid,:])
            df_illus = self.df_th
            return self.illustrator.show_trade_graph(df_illus, self.df_tf.reset_index().set_index('date'), tid, large_view=large_view)
        elif index != None:
            if rule:
                df_illus = self.df_th.set_index(['rule', 'trade_id']).sort_values("rule").loc[f'Rule {rule}'].reset_index()
            else:
                df_illus = self.df_th
            print(df_illus.loc[index, :])
            tid = df_illus.loc[index, 'trade_id']
            return self.illustrator.show_trade_graph(df_illus.set_index('trade_id'), self.df_tf.reset_index().set_index('date'), tid, large_view=large_view)

    def calculate_trading_history(self, df_tf: pd.DataFrame=None, strategy: FabStrategy=None, sensitivity: float=0.001) -> str:
        """
        Tests the asset in history, with respect to the rules outlined in the FabStrategy class.
        It adds applicable trades to a list and then an Analyzer object summarizes the profitability

        Parameters:
        -----------
        df_tf:       pd.DataFrame - The abstracted tf data that is ready for backtesting. This is not minute data if tf != 1
        strategy:    Object - any trading strategy that takes the index and sensitivity as input, and returns boolean values.
        sensitivity: float  - Allowance between price and MA. The larger the value, the further and less sensitive.


        :return str - A summary of all metrics in the backtest. (See Analyzer.summarize_statistics method for more info)
        """
        self.trade_history = TradeHistory()

        df_tf = self.df_tf if type(df_tf) == None else df_tf
        strategy = FabStrategy() if not strategy else strategy

        list_of_str_dates = [str(date) for date in df_tf['date']]

        # Creating necessary moving averages from FabStrategy class
        strategy.load_data(df_tf)
        strategy.update_moving_averages()

        # Iterating through every single data point and checking if rules apply.
        for row_index in range(231, len(df_tf) - 1):
            self.check_rule_1(strategy, row_index, list_of_str_dates)
            self.check_rule_2(strategy, row_index, list_of_str_dates, sensitivity)
            self.check_rule_3(strategy, row_index, list_of_str_dates)

        return self.trade_history

    def analyze(self):
        # Analyzing the trade history
        self.analyzer.calculate_statistics(self.trade_history)

        # Adding all trades in a list. They are in the form of: 1+profit margin. Ex. [1.04, 0.97, 1.12] etc.
        self.trades = self.analyzer.get_trades()
        self.pnl = round(self.analyzer.get_pnl(self.trades), 3)
        self.summary = self.analyzer.summarize_statistics()


    def start_backtest(self, df_th):
        self.calculate_trading_history()
        self.analyze()
        return