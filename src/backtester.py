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
from functools import reduce
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

    def __init__(self, db=True):
        self.loader = _DataLoader(db=db)
        self.analyzer = Analyzer()
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
            self.trade_history.append(Trade(["Long", "Enter", list_of_str_dates[i], strategy.black[i]*(1+sensitivity), "Rule 2"]))

        elif strategy.rule_2_short_enter(i, sensitivity) and self.trade_history.last_trade().status != "Enter":
            self.trade_history.append(Trade(["Short", "Enter", list_of_str_dates[i], strategy.black[i]*(1-sensitivity), "Rule 2"]))

        if strategy.rule_2_buy_stop_absolute(i) and self.trade_history.last_trade().rule == "Rule 2" and \
                self.trade_history.last_trade().side == "Long" and self.trade_history.last_trade().status == "Enter":
            self.trade_history.append(Trade(["Long", "Exit", list_of_str_dates[i], strategy.price[i], "Rule 2"]))

        if strategy.rule_2_short_stop_absolute(i) and self.trade_history.last_trade().rule == "Rule 2" and \
                self.trade_history.last_trade().side == "Short" and self.trade_history.last_trade().status == "Enter":
            self.trade_history.append(Trade(["Short", "Exit", list_of_str_dates[i], strategy.price[i], "Rule 2"]))

        elif strategy.rule_2_short_stop(i) and self.trade_history.last_trade().rule == "Rule 2" and \
                self.trade_history.last_trade().side == "Short" and self.trade_history.last_trade().status == "Enter":
            self.trade_history.append(Trade(["Short", "Exit", list_of_str_dates[i], strategy.price[i], "Rule 2"]))

        elif strategy.rule_2_buy_stop(i) and self.trade_history.last_trade().rule == "Rule 2" and \
                self.trade_history.last_trade().side == "Long" and self.trade_history.last_trade().status == "Enter":
            self.trade_history.append(Trade(["Long", "Exit", list_of_str_dates[i], strategy.price[i], "Rule 2"]))


    def check_rule_3(self, strategy, i, list_of_str_dates):
        if strategy.rule_3_buy_enter(i) and self.trade_history.last_trade().status != "Enter":
            self.trade_history.append(Trade(["Long", "Enter", list_of_str_dates[i+1], strategy.price[i], "Rule 3"]))

        elif strategy.rule_3_short_enter(i) and self.trade_history.last_trade().status != "Enter":
            self.trade_history.append(Trade(["Short", "Enter", list_of_str_dates[i+1], strategy.price[i], "Rule 3"]))

    def graph_trade(self, tid=None, index=None, rule=None, adjust_left_view= 29, adjust_right_view = 10, df_th = None):
        """Assuming no previous index on trading history
        Indices must also be calculated! """

        df_th = self.df_th if type(df_th) == type(None) else df_th

        if tid != None:
            df_th = df_th.set_index('tid')
            print(df_th.drop(['strategy', 'side', 'symbol', 'tf', 'candles'], axis=1).loc[tid, :])
            df_illus = df_th
            return self.illustrator.show_trade_graph(df_illus, self.df_tf.reset_index().set_index('date'), tid,
                                                     adjust_left_view=adjust_left_view, adjust_right_view=adjust_right_view)
        elif index != None:
            if rule:
                df_illus = df_th.set_index(['rule', 'tid']).sort_values("rule").loc[f'Rule {rule}'].reset_index()
            else:
                df_illus = df_th
            print(df_illus.drop(['strategy', 'side', 'symbol', 'tf', 'candles'], axis=1).loc[index, :])
            tid = df_illus.loc[index, 'tid']
            return self.illustrator.show_trade_graph(df_illus.set_index('tid'), self.df_tf.reset_index().set_index('date'), tid,
                                                     adjust_left_view=adjust_left_view, adjust_right_view=adjust_right_view)

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
        df_tf = self.df_tf if type(df_tf) == type(None) else df_tf
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

    def create_trade_history_table(self):
        """Only use this function once you complete a backtest with an instance"""

        trade_history = pd.DataFrame(
            columns=["tid", "enter_date", "exit_date", "strategy", "rule", "side", "symbol", "tf", 'enter_price',
                     'exit_price', "candles"])
        tid = 0
        for i in range(2, len(self.trade_history), 2):
            tid += 1
            enter_date = self.trade_history[i - 1].datetime
            exit_date = self.trade_history[i].datetime
            strategy = "BOSS"
            rule = self.trade_history[i - 1].rule
            side = self.trade_history[i - 1].side
            symbol = self.symbol
            tf = self.tf
            enter_price = self.trade_history[i - 1].price
            exit_price = self.trade_history[i].price
            candles = int((np.datetime64(exit_date) - np.datetime64(enter_date)) / np.timedelta64(1, 'm')) + 1

            row = pd.DataFrame(
                [[tid, enter_date, exit_date, strategy, rule, side, symbol, tf, enter_price, exit_price, candles]],
                columns=['tid', "enter_date", "exit_date", "strategy", "rule", "side", "symbol", "tf",
                         'enter_price', 'exit_price', 'candles'])
            trade_history = trade_history.append(row)

        trade_history[['enter_date', 'exit_date']] = trade_history[['enter_date', 'exit_date']].astype(np.datetime64)
        return trade_history.reset_index(drop=True)

    @staticmethod
    def bridge_table_creator(df_candles, df_th):
        df_candle_id = df_candles.reset_index()['candle_id']
        df_candle_ts = df_candles.reset_index()['timestamp']
        bridge = pd.DataFrame(columns=['tid', 'candle_id'])
        #     start = time.perf_counter()
        for tid, enter_date, exit_date, in zip(df_th['tid'], df_th['enter_date'], df_th['exit_date']):
            enter_timestamp = int(pd.Timestamp(enter_date, tz='utc').timestamp())
            exit_timestamp = int(pd.Timestamp(exit_date, tz='utc').timestamp())

            right = df_candle_id[df_candle_ts.between(enter_timestamp, exit_timestamp)].reset_index(drop=True)
            left = pd.Series(data=[tid for _ in range(len(right))], name='tid', dtype='int64')

            bridge = bridge.append(pd.concat([left, right], axis=1))

        #     print(f"All Done: {time.perf_counter()-start}")
        return bridge.reset_index(drop=True)

    def generate_detailed_trading_history(self, df_candles=None, df_th=None):
        df_candles = self.df if type(df_candles) == type(None) else df_candles
        df_th = self.df_th if type(df_th) == type(None) else df_th

        bridge = self.bridge_table_creator(df_candles, df_th)
        df_th_merge = df_th[['tid', 'symbol', 'tf', 'enter_price', 'exit_price', 'strategy', 'rule', 'side']]
        df_candles_merge = df_candles[['timestamp', 'date', 'open', 'high', 'low', 'close', 'volume']]

        dfs_to_merge = [bridge, df_th_merge]

        detailed_th_partial = reduce(lambda left, right: pd.merge(left, right, how ='left', on='tid'), dfs_to_merge)
        detailed_th = detailed_th_partial.merge(df_candles_merge, on='candle_id')
        return detailed_th

    def add_metrics_to_trading_history(self, df_th, analyzer = None):
        analyzer = self.analyzer if type(analyzer) == type(None) else analyzer

        peak_index_merge = self.analyzer.peak_index[['peak_price', 'trough_price']]
        trade_index_merge = self.analyzer.trade_index['profitability']
        peak_profit_merge = analyzer.get_peak_unrealized_profit(analyzer.trade_index, analyzer.peak_index, 0.9996, median=False).drop('pnl', axis=1)
        peak_loss_merge = analyzer.get_peak_unrealized_loss(analyzer.trade_index, analyzer.peak_index, 0.9996, median=False).drop('pnl',axis=1)

        dfs_to_merge = [df_th, peak_index_merge, trade_index_merge, peak_profit_merge, peak_loss_merge]

        df_th_extra = reduce(lambda left, right: pd.merge(left, right, how='left', on='tid'), dfs_to_merge)
        self.df_th = df_th_extra
        return self.df_th

    def generate_asset_metrics(self, detailed_th, rule):
        """Input either 'Long' Detailed Trading History or 'Short' Detailed Trading History"""
        rule_detailed_th = detailed_th.reset_index().set_index(['rule', 'tid', 'candle_id'])
        try:
            self.analyzer.generate_all_indices(rule_detailed_th.loc[rule])
        except KeyError:
            return None

        symbol = self.symbol
        timeframe = self.tf
        side = detailed_th['side'].iloc[0]
        analyzer = self.analyzer

        candle_index     = analyzer.candle_index
        peak_index       = analyzer.peak_index
        volume_index     = analyzer.candle_volume_index
        pps_index        = analyzer.candle_pps_index
        volatility_index = analyzer.candle_volatility_index
        trade_index      = analyzer.trade_index

        profit_rate      = analyzer.get_profit_rate(candle_index, mean=True)
        loss_rate        = analyzer.get_loss_rate(candle_index, mean=True)
        peak_profit_rate = analyzer.get_peak_profit_rate(candle_index, peak_index, mean=True)
        peak_loss_rate   = analyzer.get_peak_loss_rate(candle_index, peak_index, mean=True)
        pps_rate         = analyzer.get_pps_rate(pps_index, median=True)
        peak_pps_rate    = analyzer.get_peak_pps_rate(pps_index, peak_index, mean=True)
        volatility_rate  = analyzer.get_volatility_rate(volatility_index, median=True)
        volume_rate      = analyzer.get_volume_rate(volume_index, median=True)
        peak_volume_rate = analyzer.get_peak_volume_rate(volume_index, peak_index, mean=True)

        num_candles_to_peak    = analyzer.get_num_candles_to_peak(candle_index, peak_index, median=True)
        peak_unrealized_profit = analyzer.get_peak_unrealized_profit(trade_index, peak_index, 0.9996, median=True)
        peak_unrealized_loss   = analyzer.get_peak_unrealized_loss(trade_index, peak_index, 0.9996, median=True)
        volume                 = analyzer.get_volume(volume_index, median=True)
        peak_volume            = analyzer.get_peak_volume(volume_index, peak_index, median=True)
        profitability          = analyzer.calculate_profitability(trade_index, median=True)

        largest_profit  = analyzer.get_largest_profit(trade_index)
        largest_loss    = analyzer.get_largest_loss(trade_index)
        gross_profit    = analyzer.get_gross_profit(trade_index)
        gross_loss      = analyzer.get_gross_loss(trade_index)
        num_trades_won  = analyzer.get_num_trades_won(trade_index)
        num_trades_lost = analyzer.get_num_trades_lost(trade_index)

        average_win     = analyzer.get_average_win(gross_profit, num_trades_won)
        average_loss    = analyzer.get_average_loss(gross_loss, num_trades_lost)


        peak_unrealized_profit_index = analyzer.get_peak_unrealized_profit(trade_index, peak_index, 0.9996, median=False)['peak_profit']
        peak_unrealized_loss_index   = analyzer.get_peak_unrealized_loss(trade_index, peak_index, 0.9996, median=False)['peak_loss']
        unrealized_rrr               = analyzer.get_unrealized_rrr(peak_unrealized_profit_index, peak_unrealized_loss_index)
        average_rrr                  = analyzer.get_average_rrr(average_win, average_loss)
        amount_of_data               = analyzer.get_amount_of_data(trade_index)

        df_metrics = pd.DataFrame([[symbol, timeframe, rule, side, unrealized_rrr, peak_unrealized_profit,
                                    num_candles_to_peak, peak_volume, amount_of_data, average_rrr, profitability,
                                    average_win, peak_unrealized_loss, average_loss, volume, largest_profit,
                                    profit_rate,loss_rate, peak_profit_rate, peak_loss_rate, pps_rate,
                                    peak_pps_rate, volatility_rate, volume_rate, peak_volume_rate, largest_loss]],

                            columns=['symbol', 'tf', 'rule_no.', 'side', "unrealized rrr", "peak unreal. profit",
                                     "num candles to peak", "peak volume", "amount of data", "average rrr",
                                     "profitability", "average win", "peak unreal. loss", "average loss", "volume",
                                     "largest profit", "profit rate", "loss rate", "peak profit rate", "peak loss rate",
                                     "pps rate", "peak pps rate", "volatility rate", "volume rate", "peak volume rate",
                                     "largest loss"])
        self.df_metrics = df_metrics.set_index(['symbol', 'tf', 'rule_no.'])
        return self.df_metrics

    def start_backtesting(self, tf=None):
        tf = self.tf if not tf else tf
        self.set_timeframe(tf)
        self.df_tf = self.load_timeframe_data()
        self.calculate_trading_history()
        self.df_th = self.create_trade_history_table()
        self.detailed_th = self.generate_detailed_trading_history(self.df).set_index(['tid', 'candle_id'])
        # self.df_th = self.add_metrics_to_trading_history(self.df_th)
        return self

    def collect_historical_metrics(self, symbol_list, tf_list):
        df_metrics_master = pd.DataFrame()
        for symbol in symbol_list:
            print(f"Working on: {symbol}")
            b = Backtester()
            b.set_symbol(symbol)
            b.load_backtesting_data(all_data=True)
            for tf in tf_list:
                print(f"\t Working on: {tf}m")
                b2 = Backtester(db=False)
                b2.symbol = b.symbol
                b2.df = b.df
                b2.start_backtesting(tf=tf)
                for rule in ['Rule 1', 'Rule 2', 'Rule 3']:
                    short_detailed_th = b2.detailed_th[b2.detailed_th['side'] == 'Short']
                    long_detailed_th = b2.detailed_th[b2.detailed_th['side'] == 'Long']
                    long_metrics = b2.generate_asset_metrics(long_detailed_th, rule)
                    short_metrics = b2.generate_asset_metrics(short_detailed_th, rule)
                    df_metrics_master = df_metrics_master.append(long_metrics)
                    df_metrics_master = df_metrics_master.append(short_metrics)
            b.loader.conn.close()
        return df_metrics_master.set_index('side', append=True)

