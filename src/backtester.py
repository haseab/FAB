import os
import random
import time
from datetime import datetime, timedelta
from functools import reduce
from threading import Thread
from typing import Union

import numpy as np
import pandas as pd
from IPython.display import clear_output, display

from analyzer import Analyzer
from dataloader import _DataLoader
from fab_strategy import FabStrategy
from helper import Helper
from illustrator import Illustrator
from trade import Trade
from trading_history import TradeHistory


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

    def set_date_range(self, start_date: str, end_date: str = None) -> Union[str, str]:
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

    def load_backtesting_data(self, symbol=None, start_date=None, end_date=None, table_name="candlesticks", all_data=True):
        cursor = self.loader.sql.conn.cursor()
        symbol = self.symbol if not symbol else symbol
        self.symbol = symbol

        all_data = False if start_date else True

        if all_data:
            df = self.loader.sql.SELECT(f"* FROM {table_name} WHERE SYMBOL = '{symbol}' AND TF = '1'", cursor)
        else:
            start_date = self.start_date if not start_date else start_date
            end_date = str(datetime.now())[:19] if not end_date else end_date
            df = self.loader.sql.SELECT(f"* FROM {table_name} WHERE SYMBOL = '{symbol}' AND TF = '1' AND DATE "
                                        f"BETWEEN '{start_date}' AND '{end_date}'", cursor)
                                    
        # df = df.drop('id', axis=1)
        df = df.sort_values('timestamp')
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
        last_trade = self.trade_history.last_trade()

        if strategy.rule_1_buy_enter(i) and last_trade.status != "Enter":
            self.trade_history.append([i+1,"Long", "Enter", list_of_str_dates[i+1], strategy.close[i], "Rule 1"])

        elif strategy.rule_1_buy_exit(i) and last_trade.side == "Long" and last_trade.status == "Enter":
            self.trade_history.append([i+1,"Long", "Exit", list_of_str_dates[i+1], strategy.close[i], "Rule 1"])

        elif strategy.rule_1_short_enter(i) and last_trade.status != "Enter":
            self.trade_history.append([i+1, "Short", "Enter", list_of_str_dates[i+1], strategy.close[i], "Rule 1"])

        elif strategy.rule_1_short_exit(i) and last_trade.side == "Short" and last_trade.status == "Enter":
            self.trade_history.append([i+1, "Short", "Exit", list_of_str_dates[i+1], strategy.close[i], "Rule 1"])

    def check_rule_2(self, strategy, i, list_of_str_dates, v2=False):
        last_trade = self.trade_history.last_trade()

        if v2:
            strategy.rule_2_short_enter = strategy.rule_2_short_enter_v2
            strategy.rule_2_buy_enter = strategy.rule_2_buy_enter_v2

        if strategy.rule_2_buy_enter(i) and last_trade.status != "Enter":
            self.trade_history.append([i+1, "Long", "Enter", list_of_str_dates[i+1], strategy.close[i]*(1+strategy.allowance), "Rule 2"])

        elif strategy.rule_2_short_enter(i) and last_trade.status != "Enter":
            self.trade_history.append([i+1, "Short", "Enter", list_of_str_dates[i+1], strategy.close[i]*(1-strategy.allowance), "Rule 2"])

        elif strategy.rule_2_buy_stop_absolute(i) and last_trade.rule == "Rule 2" and last_trade.side == "Long" and last_trade.status == "Enter":
            self.trade_history.append([i+1, "Long", "Exit", list_of_str_dates[i+1], strategy.close[i], "Rule 2"])

        elif strategy.rule_2_short_stop_absolute(i) and last_trade.rule == "Rule 2" and last_trade.side == "Short" and last_trade.status == "Enter":
            self.trade_history.append([i+1, "Short", "Exit", list_of_str_dates[i+1], strategy.close[i], "Rule 2"])

        elif strategy.rule_2_short_stop(i) and last_trade.rule == "Rule 2" and last_trade.side == "Short" and last_trade.status == "Enter":
            self.trade_history.append([i+1, "Short", "Exit", list_of_str_dates[i+1], strategy.close[i], "Rule 2"])

        elif strategy.rule_2_buy_stop(i) and last_trade.rule == "Rule 2" and last_trade.side == "Long" and last_trade.status == "Enter":
            self.trade_history.append([i+1, "Long", "Exit", list_of_str_dates[i+1], strategy.close[i], "Rule 2"])

    def check_rule_3(self, strategy, i, list_of_str_dates):
        last_trade = self.trade_history.last_trade()

        if strategy.rule_3_buy_enter(i) and last_trade.status != "Enter":
            self.trade_history.append([i+1, "Long", "Enter", list_of_str_dates[i+1], strategy.close[i], "Rule 3"])

        elif strategy.rule_3_short_enter(i) and last_trade.status != "Enter":
            self.trade_history.append([i+1, "Short", "Enter", list_of_str_dates[i+1], strategy.close[i], "Rule 3"])

    def graph_trade(self, tid=None, index=None, rule=None, adjust_left_view=150, adjust_right_view=10, df_th=None,
                    tf=None, test_df=False, flat=False, save=False, data_only=False):
        """Assuming no previous index on trading history
        Indices must also be calculated! """

        df_th = self.df_th if type(df_th) == type(None) else df_th
        tf = self.tf if not tf else tf
        df_tf = self.df_tf
        # test_df = False if tf == self.tf else test_df

        if index != None:
            if rule:
                df_illus = df_th.set_index(['rule', 'tid']).sort_values("rule").loc[f'Rule {rule}'].reset_index()
            else:
                df_illus = df_th
            tid = df_illus.loc[index, 'tid']

        if self.tf != tf:
            start_datetime = df_th.set_index('tid').loc[tid, 'enter_date'] - pd.Timedelta((231 * tf) + adjust_left_view * tf, 'minutes')
            end_datetime = df_th.set_index('tid').loc[tid, 'exit_date'] + pd.Timedelta((adjust_right_view * tf), 'minutes')
            trade_enter_datetime = df_th.set_index('tid').loc[tid, 'enter_date'] + pd.Timedelta(self.tf-1 + self.tf*adjust_right_view, 'minutes')

            # raise Exception(start_datetime, trade_enter_datetime, trade_enter_datetime_2)
            df = self.df.reset_index()
            trade_df = df[df['date'].between(start_datetime, end_datetime)].copy()
            df_tf = self.loader._timeframe_setter(trade_df, tf, keep_last_row=True).reset_index()

            if test_df:
                pre_trade_df = df[df['date'].between(start_datetime, trade_enter_datetime)].copy()
                pre_trade_df_tf = self.loader._timeframe_setter(pre_trade_df, tf, keep_last_row=True).reset_index()
                df_graph = self.illustrator.add_sma_to_df(pre_trade_df_tf).set_index('date')

                self.new = pre_trade_df

                if data_only:
                    return df_graph
                return self.illustrator.graph_df(df_graph, flat=flat)


        if tid != None:
            df_th = df_th.set_index('tid')
            if not data_only and not test_df:
                print(df_th.drop(['side', 'symbol', 'tf', 'candles'], axis=1).loc[tid, :])
            df_illus = df_th
            return self.illustrator.prepare_trade_graph_data(df_illus, df_tf, tid, tf=tf, data_only=data_only, flat=flat,
                                                     adjust_left_view=adjust_left_view, save=save, adjust_right_view=adjust_right_view)
        if index != None:
            if not data_only and not test_df:
                print(df_illus.drop(['side', 'symbol', 'tf', 'candles'], axis=1).loc[index, :])
            return self.illustrator.prepare_trade_graph_data(df_illus.set_index('tid'), df_tf, tid, data_only=False, flat=flat,
                                                     tf=tf, adjust_left_view=adjust_left_view, save=save, adjust_right_view=adjust_right_view)

    def graph_benchmark(self, object, enter_datetime, adjust_left_view, base_tf, adjust_right_view=0, space=50,
                        description=True, flat=False, tf=None, exit_datetime=None, btc=False):
        if description:
            print(f"Symbol: {object.symbol}")
        tf = base_tf if not tf else tf

        start_datetime = enter_datetime - pd.Timedelta((231 * tf) + adjust_left_view * tf, 'minutes')
        trade_enter_datetime = enter_datetime + pd.Timedelta(base_tf - 1 + adjust_right_view, 'minutes')
        df_object = object.df[object.df['date'].between(start_datetime, trade_enter_datetime)].copy()
        if exit_datetime:
            exit_datetime += + pd.Timedelta(10*base_tf, 'minutes')
            trade_enter_datetime = exit_datetime
            df_object = object.df[object.df['date'].between(start_datetime, exit_datetime)].copy()
        df_object_tf = self.loader._timeframe_setter(df_object, tf, keep_last_row=True).reset_index()
        # print(df_object_tf)

        if len(df_object_tf) < 280:
            clear_output(wait=True)
            print('Not enough data to cover timeframe, returning to original timeframe.....')
            df_object_tf = self.graph_benchmark(self, enter_datetime, adjust_left_view, base_tf, adjust_right_view, space, flat)
            return df_object_tf

        if btc:
            print(f"Symbol: {self.symbol[:-4] + object.symbol[:3]}")
            df_symbol = self.df[self.df['date'].between(start_datetime, trade_enter_datetime)].copy()
            df_symbol_tf = self.loader._timeframe_setter(df_symbol, tf, keep_last_row=True).reset_index()
            df_btc = df_symbol_tf[['open', 'high', 'low', 'close']] / df_object_tf[['open', 'high', 'low', 'close']]
            df_btc[['symbol', 'tf', 'timestamp', 'date']] = df_symbol_tf[['symbol', 'tf', 'timestamp', 'date']]
            # print(df_object_tf, df_btc)

            self.new = df_object_tf, df_btc, df_symbol_tf

            self.illustrator.graph_df(df_btc, flat=flat, space=space)

        self.illustrator.graph_df(df_object_tf, flat=flat, space=space)
        return df_object_tf

    # def continuously_update(self, input, refresh_rate):
    #     while True:
    #         time.sleep(refresh_rate)
    #         df_graph, df_bench = self.test_intuition_output(adjust_left_view, adjusted_tf, base_tf, benchmark,
    #                                                         enter_datetime, flat, i, shift, tid, tids)
    #         # if input in ['s', 'stop', 'STOP', 'S']:
    #         #     return

    def download_all_trades(self, rule=None, tid_list=None, save=False, results=False, space=50):
        # df_tf_candles = df_tf_candles.reset_index().set_index('date')
        df_th = self.df_th.reset_index().set_index('tid')

        if rule:
            df_th = df_th[df_th['rule'] == f'Rule {rule}']

        tids = tid_list if tid_list else df_th.index

        for tid in tids:
            enter_datetime = df_th.loc[tid, 'enter_date']
            if results:
                self.graph_trade(tid, save=save)
                print(f'saved file:{tid}')
            else:
                df_graph = self.graph_trade(tid=tid, data_only=True).reset_index()
                self.illustrator.graph_df(df_graph[df_graph['date'] <= enter_datetime].set_index('date'), save=save,
                                          space=space, tid=tid)
        return "Saved all of them"

    def test_intuition_output(self, adjust_left_view, adjusted_tf, base_tf, benchmark, enter_datetime, flat,
                              i, shift, tid, tids, btc=False, descriptions=True):
        df_bench = None
        clear_output(wait=True)
        if descriptions:
            print(f'Answer: {shift}')
            print(f"Trade {i} of {len(tids)} ")
            print(f"Trade id: {tid}")
        df_graph = self.graph_benchmark(self, enter_datetime, flat=flat, adjust_left_view=adjust_left_view,
                                        adjust_right_view=shift, base_tf=base_tf, description=descriptions, tf=adjusted_tf)
        if type(benchmark) != type(None):
            df_bench = self.graph_benchmark(benchmark, enter_datetime, flat=flat, adjust_left_view=adjust_left_view,
                                 adjust_right_view=shift, base_tf=base_tf, description= descriptions, btc=btc, tf=adjusted_tf)
        return df_graph, df_bench
    
    def optimal_tf_calculator(self, symbols_list, tf_list, delay_list):
        df = pd.DataFrame()
        for symbol in symbols_list:
            self.loader = _DataLoader(db=True)
            self.load_backtesting_data(symbol=symbol, all_data=True)
            if len(self.df) == 0:
                continue
            for tf in tf_list:
                for delay in delay_list:
                    print(symbol, tf, delay)
                    self.start_backtesting(tf, delay=delay)
                    if len(self.trade_history) in [0,1,2]:
                        continue
                    for rule in [1, 2, 3]:
                        for side in ['Long', 'Short']:
                            df = pd.concat([df, self.generate_asset_metrics(self.detailed_th, f'Rule {rule}', side, delay=delay)])
            print('saved', symbol)
            df.to_csv(f"Full TF Delay Backtest {symbols_list[0]} to {symbols_list[-1]}.csv")
        return df

    def optimal_delay_calculator(self, symbol_list, delay_list, tf, rule, side):
        df = pd.DataFrame()
        for symbol in symbol_list:
            self.loader = _DataLoader(db=True)
            self.load_backtesting_data(symbol=symbol, all_data=True)
            for delay in delay_list:
                print(symbol, delay)
                self.start_backtesting(tf=tf, delay=delay)
                row = self.generate_asset_metrics(self.detailed_th, f'Rule {rule}', side, delay=delay)
                df = df.append(row)
        return df

    def optimal_sensitivity_calculator(self, symbol_list, sensitivity_list, tf, delay, rule, side):
        df = pd.DataFrame()
        for symbol in symbol_list:
            self.loader = _DataLoader(db=True)
            self.load_backtesting_data(symbol=symbol, all_data=True)
            for sensitivity in sensitivity_list:
                print(symbol, sensitivity)
                self.start_backtesting(tf=tf, delay=delay, sensitivity=sensitivity)
                row = self.generate_asset_metrics(self.detailed_th, f'Rule {rule}', side, sensitivity=sensitivity)
                df = df.append(row)
        return df

    def test_your_intuition(self, df_th, limit=100, offset=0, adjust_left_view=200, adjust_right_view=250, show_answers=False, flat=True,
                            tids=None, benchmark=None, btc=False, refresh_rate=0.1, skip_rate=1, save=False, space=50):
        df_th = df_th.set_index('tid')
        self.my_trades = []

        base_tf = df_th['tf'].iloc[0]
        adjusted_tf = None
        rule = df_th['rule'].iloc[0]

        random.seed(1)
        if not tids:
            self.all_trades = list(df_th.index[offset:offset+limit])
            random.shuffle(self.all_trades)
            tids = self.all_trades
        # self.all_trades = self.all_trades[offset:offset+limit]

        for i, tid in enumerate(tids):
            enter_datetime = self.df_th.set_index('tid')['enter_date'].loc[tid]
            exit_datetime = self.df_th.set_index('tid')['exit_date'].loc[tid]
            clear_output(wait=True)
            i += 1
            shift = 0

            print(f"Trade {i} of {len(tids)} ")
            print(f"Trade id: {tid}")
            df_graph = self.graph_trade(tid=tid, adjust_left_view=adjust_left_view,df_th=df_th.reset_index(),
                                        adjust_right_view=adjust_right_view, data_only=True).reset_index()

            # self.illustrator.graph_df(df_graph[df_graph['date'] < enter_datetime + 8*timedelta(minutes=base_tf)].set_index('date'), space=space, flat=flat)
            if rule == 2:
                self.illustrator.graph_df(df_graph[df_graph['date'] <= enter_datetime].set_index('date'), space=space, flat=flat)
            else:
                self.illustrator.graph_df(df_graph[df_graph['date'] < enter_datetime].set_index('date'), space=space, flat=flat)


            if type(benchmark) != type(None):
                self.graph_benchmark(benchmark, enter_datetime, flat=flat, adjust_left_view=adjust_left_view, base_tf=base_tf, btc=btc)

            while True:
                answer = input('Would you make this trade? y/n: ')

                if answer == '..':
                    count = 0
                    adjusted_tf = adjusted_tf if adjusted_tf else df_graph['tf'].iloc[0]
                    while True:
                        count += skip_rate
                        time.sleep(refresh_rate)
                        df_graph, df_bench = self.test_intuition_output(adjust_left_view, adjusted_tf, base_tf, benchmark,
                                                                        enter_datetime, flat, i, count, tid, tids, btc=btc, descriptions=False)

                elif answer[:2] == '..' and len(answer) > 2:
                    shift = int(answer[2:])
                    adjusted_tf = adjusted_tf if adjusted_tf else df_graph['tf'].iloc[0]

                    df_graph, df_bench = self.test_intuition_output(adjust_left_view, adjusted_tf, base_tf, benchmark,
                                                          enter_datetime, flat, i, shift, tid, tids, btc=btc)
                elif answer[:1] == 'n' and len(answer) > 1:
                    if not answer[1:]:
                        shift += base_tf
                    elif answer[1:].strip('-').isdigit():
                        shift += int(answer[1:])
                    df_graph, df_bench = self.test_intuition_output(adjust_left_view, adjusted_tf, base_tf, benchmark,
                                                          enter_datetime, flat, i, shift, tid, tids, btc=btc)
                elif answer == 'y':
                    self.my_trades.append(tid)
                    break
                elif answer == 'n':
                    break
                elif answer.isdigit() and int(answer) < 1500:
                    adjusted_tf = int(answer)

                    df_graph, df_bench = self.test_intuition_output(adjust_left_view, adjusted_tf, base_tf, benchmark,
                                                          enter_datetime, flat, i, shift, tid, tids, btc=btc)

                elif answer == 'saveoff':
                    save = False
                elif answer == 'saveon':
                    save = True
                else:
                    print('no valid answer entered, try again....')
                    continue
            if show_answers:
                clear_output(wait=True)
                print(f"Trade {i} of {len(tids)} ")
                print(f"Trade id: {tid}")
                profit = round(df_th.loc[tid, 'exit_price'] / df_th.loc[tid, 'enter_price'], 5)
                print(f"Profit: {profit}")
                self.graph_trade(tid=tid, rule=rule, adjust_left_view=adjust_left_view, test_df=True, df_th=df_th.reset_index(), flat=flat, data_only=False)

                if type(benchmark) != type(None):
                    self.graph_benchmark(benchmark, enter_datetime, flat=flat, adjust_left_view=adjust_left_view, base_tf=base_tf, space=0, exit_datetime=exit_datetime, btc=btc)

                while True:
                    resp = input('Press c to continue... ')

                    if resp.isdigit() and int(resp)<1500:
                        adjusted_tf = int(resp)
                        clear_output(wait=True)
                        self.graph_trade(tid=tid, rule=rule, adjust_left_view=adjust_left_view, test_df=False, tf=adjusted_tf, df_th=df_th.reset_index(), flat=flat, data_only=False)

                    elif resp == 'c' or ' ':
                        break

        self.trade_disparity = pd.DataFrame()
        self.trade_disparity['tid'] = tids
        self.trade_disparity['my trades'] = [True if tid in self.my_trades else False for tid in self.trade_disparity['tid']]
        self.trade_disparity = self.trade_disparity.merge(self.analyzer.trade_index[['profitability']], on='tid')

        if save:
            symbol = df_th['symbol'].iloc[0]
            tids = sorted(tids)
            self.all_trades = tids
            first_tid = tids[0]
            last_tid = tids[-1]
            if type(benchmark) == type(None):
                self.trade_disparity.to_csv(f'Intuition test {symbol} {base_tf}m, Rule #{rule}, tids {first_tid}-{last_tid}.csv', index=False)
            else:
                self.trade_disparity.to_csv(f'Intuition test {symbol} {base_tf}m with benchmark, Rule #{rule}, tids {first_tid}-{last_tid}.csv', index=False)

        theoretical_profitability = self.trade_disparity['profitability'].product()
        actual_profitability = self.trade_disparity[self.trade_disparity['my trades']]['profitability'].product()

        return theoretical_profitability, actual_profitability

    def calculate_trading_history(self, df_tf: pd.DataFrame=None, strategy: FabStrategy=None, delay=0, sensitivity=0, v2=False) -> str:
        """
        Tests the asset in history, with respect to the rules outlined in the FabStrategy class.
        It adds applicable trades to a list and then an Analyzer object summarizes the profitability

        Parameters:
        -----------
        df_tf:       pd.DataFrame - The abstracted tf data that is ready for backtesting. This is not minute data if tf != 1
        strategy:    Object - any trading strategy that takes the index and allowance as input, and returns boolean values.
        allowance: float  - Allowance between price and MA. The larger the value, the further and less sensitive.


        :return str - A summary of all metrics in the backtest. (See Analyzer.summarize_statistics method for more info)
        """
        self.trade_history = TradeHistory()
        df_tf = self.df_tf if type(df_tf) == type(None) else df_tf
        strategy = FabStrategy(delay=delay, sensitivity=sensitivity) if not strategy else strategy
        list_of_str_dates = [str(date) for date in df_tf['date']]

        # Creating necessary moving averages from FabStrategy class
        strategy.load_data(df_tf)
        # Iterating through every single data point and checking if rules apply.
        for row_index in range(231, len(df_tf) - 1):
            self.check_rule_1(strategy, row_index, list_of_str_dates)
            self.check_rule_2(strategy, row_index, list_of_str_dates, v2=v2)
            self.check_rule_3(strategy, row_index, list_of_str_dates)

        return self.trade_history

    def create_trade_history_table(self, trade_history_list):
        if len(trade_history_list) in [0,1,2]:
            return pd.DataFrame(columns=['tid', 'enter_index', 'exit_index', 'enter_date', 'exit_date', 'rule', 'side', 
                                        'symbol', 'tf', 'enter_price', 'exit_price', 'candles'])

        test_th = pd.DataFrame(np.array(trade_history_list[1:]), columns=['index', 'side', 'enter', 'date', 'price', 'rule'])

        initial_columns = ['index', 'side', 'enter', 'date', 'price', 'rule']
        enter_columns = ['enter_index', 'side', 'enter', 'enter_date', 'enter_price', 'rule']
        exit_columns = ['exit_index', 'side', 'exit', 'exit_date', 'exit_price', 'rule']

        df_enter = test_th[::2].rename(columns = dict(zip(initial_columns, enter_columns))).reset_index(drop=True).drop('enter', axis=1)
        df_enter['tid'] = range(1,len(df_enter)+1)

        df_exit = test_th[1::2].rename(columns = dict(zip(initial_columns, exit_columns))).reset_index(drop=True).drop(['side', 'rule', 'exit'], axis=1)
        df_exit['tid'] = range(1,len(df_exit)+1)

        if len(df_enter) == len(df_exit) + 1:
            df_enter = df_enter.drop(df_enter.tail(1).index)

        elif len(df_enter) > len(df_exit) + 1:
            raise("issue with size of dfs")

        df = df_enter.merge(df_exit, on='tid')

        df['symbol'] = [self.symbol]*len(df_enter)
        df['tf'] = [self.tf]*len(df_enter)

        df[['enter_date','exit_date']] = df[['enter_date','exit_date']].astype('datetime64')
        df[['enter_index','exit_index', 'tf']] = df[['enter_index','exit_index', 'tf']].astype('int64')
        df[['enter_price','exit_price']] = df[['enter_price','exit_price']].astype('float64').round(5)
        
        self.test = df
        df['candles'] = df['exit_index']-df['enter_index'] + 1
        # df['candles'] = ((df['exit_date'] - df['enter_date']) / np.timedelta64(self.tf, 'm') + 1).astype(int)
        df = df[['tid', 'enter_index', 'exit_index', 'enter_date', 'exit_date', 'rule', 'side', 'symbol', 'tf', 'enter_price', 'exit_price', 'candles']]
        return df

    def bridge_table_creator(self, df_candles, df_th):
        df_candle_id = df_candles.reset_index()['candle_id']
        bridge = pd.DataFrame(columns=['tid', 'candle_id'])

        left_array, right_array = np.array([]), np.array([])
    
        for tid, enter_index, exit_index, in zip(df_th['tid'], df_th['enter_index'], df_th['exit_index']):

            right = df_candle_id.iloc[enter_index:exit_index+1].reset_index(drop=True).values
            right_array = np.append(right_array, right)
            left_array = np.append(left_array, np.repeat(tid, len(right)))

        bridge = bridge.append(pd.concat([pd.Series(left_array, name='tid'), pd.Series(right_array, name='candle_id')], axis=1))
        bridge = bridge.astype(int)
            
        self.bridge = bridge.reset_index(drop=True)
        return bridge.reset_index(drop=True)

    def generate_detailed_trading_history(self, df_candles, df_th):
        if len(df_th) == 0:
            return pd.DataFrame(columns=['tid', 'candle_id'])

        df_candles = df_candles.reset_index().set_index('candle_id')
        bridge = self.bridge_table_creator(df_candles, df_th)

        df_th_merge = df_th[['tid', 'symbol', 'tf', 'enter_price', 'exit_price', 'rule', 'side', 'candles']]
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

    def generate_asset_metrics(self, detailed_th, rule, side='Long', delay=0):
        """Input either 'Long' Detailed Trading History or 'Short' Detailed Trading History"""
        side_detailed_th = detailed_th[detailed_th['side'] == side]
        rule_side_detailed_th = side_detailed_th.reset_index().set_index(['rule', 'tid', 'candle_id'])
        try:
            self.analyzer.generate_all_indices(rule_side_detailed_th.loc[rule])
        except KeyError:
            return None

        symbol = self.symbol
        timeframe = self.tf
        analyzer = self.analyzer

        candle_index     = analyzer.candle_index
        peak_index       = analyzer.peak_index
        volume_index     = analyzer.candle_volume_index
        pps_index        = analyzer.candle_pps_index
        volatility_index = analyzer.candle_volatility_index
        trade_index      = analyzer.trade_index

        profit_trade_index = trade_index[trade_index['profitability'] > 1]
        profit_candle_index = profit_trade_index[[]].merge(candle_index.reset_index(), on='tid').set_index(['tid', 'candle_id'])
        profit_volume_index = profit_trade_index[[]].merge(volume_index.reset_index(), on='tid').set_index(['tid', 'candle_id'])
        profit_volatility_index = profit_trade_index[[]].merge(volatility_index.reset_index(), on='tid').set_index(['tid', 'candle_id'])
        profit_pps_index = profit_trade_index[[]].merge(pps_index.reset_index(), on='tid').set_index(['tid', 'candle_id'])
        profit_peak_index = profit_trade_index[[]].merge(peak_index.reset_index(), on='tid').set_index(['tid'])

        profit_rate      = analyzer.get_profit_rate(profit_candle_index, mean=True)
        loss_rate        = analyzer.get_loss_rate(profit_candle_index, mean=True)
        peak_profit_rate = analyzer.get_peak_profit_rate(profit_candle_index, profit_peak_index, mean=True)
        peak_loss_rate   = analyzer.get_peak_loss_rate(profit_candle_index, profit_peak_index, mean=True)
        pps_rate         = analyzer.get_pps_rate(profit_pps_index, median=True)
        peak_pps_rate    = analyzer.get_peak_pps_rate(profit_pps_index, profit_peak_index, mean=True)
        volatility_rate  = analyzer.get_volatility_rate(profit_volatility_index, median=True)
        volume_rate      = analyzer.get_volume_rate(profit_volume_index, median=True)
        peak_volume_rate = analyzer.get_peak_volume_rate(profit_volume_index, profit_peak_index, mean=True)

        num_candles_to_peak    = analyzer.get_num_candles_to_peak(profit_candle_index, profit_peak_index, median=True)
        peak_unrealized_profit = analyzer.get_peak_unrealized_profit(profit_trade_index, profit_peak_index, 0.9996, median=True)
        peak_unrealized_loss   = analyzer.get_peak_unrealized_loss(profit_trade_index, profit_peak_index, 0.9996, median=True)
        volume                 = analyzer.get_volume(profit_volume_index, median=True)
        peak_volume            = analyzer.get_peak_volume(profit_volume_index, profit_peak_index, median=True)

        profitability          = analyzer.calculate_profitability(trade_index, median=True)
        largest_profit  = analyzer.get_largest_profit(trade_index)
        largest_loss    = analyzer.get_largest_loss(trade_index)
        gross_profit    = analyzer.get_gross_profit(trade_index)
        gross_loss      = analyzer.get_gross_loss(trade_index)
        num_trades_won  = analyzer.get_num_trades_won(trade_index)
        num_trades_lost = analyzer.get_num_trades_lost(trade_index)

        win_loss_rate    = analyzer.get_win_loss_rate(num_trades_won, num_trades_lost)
        average_win     = analyzer.get_average_win(gross_profit, num_trades_won)
        average_loss    = analyzer.get_average_loss(gross_loss, num_trades_lost)

        longest_run, r_count         = analyzer.calculate_longest_run(trade_index)
        longest_drawdown, d_count    = analyzer.calculate_longest_drawdown(trade_index)
        peak_unrealized_profit_index = analyzer.get_peak_unrealized_profit(profit_trade_index, profit_peak_index, 0.9996, median=False)['peak_profit']
        peak_unrealized_loss_index   = analyzer.get_peak_unrealized_loss(profit_trade_index, profit_peak_index, 0.9996, median=False)['peak_loss']
        unrealized_rrr               = analyzer.get_unrealized_rrr(peak_unrealized_profit_index, peak_unrealized_loss_index)
        average_rrr                  = analyzer.get_average_rrr(average_win, average_loss)
        amount_of_data               = analyzer.get_amount_of_data(trade_index)
        avg_pnl                      = profitability**(1/amount_of_data)

        df_metrics = pd.DataFrame([[symbol, timeframe, delay, rule, side, amount_of_data, win_loss_rate, profitability, avg_pnl,
                                    longest_drawdown, longest_run, average_rrr, unrealized_rrr, num_candles_to_peak, average_win, peak_unrealized_loss, 
                                    average_loss, volume, largest_profit, profit_rate,loss_rate, peak_profit_rate, peak_loss_rate, pps_rate, peak_unrealized_profit,
                                    peak_pps_rate, volatility_rate, volume_rate, peak_volume, peak_volume_rate, largest_loss]],

                            columns=['symbol', 'tf', 'delay', 'rule_no.', 'side', "amount of data", "win loss rate", "profitability", 'avg pnl', 
                                     "longest drawdown", "longest_run", "average rrr", "unrealized rrr", "num candles to peak", "average win", "peak unreal. loss", 
                                     "average loss", "volume", "largest profit", "profit rate", "loss rate", "peak profit rate", "peak loss rate", "pps rate", 
                                     "peak unreal. profit", "peak pps rate", "volatility rate", "volume rate", "peak volume", "peak volume rate", "largest loss"])
        # df_metrics['avg pnl'] = df_metrics['profitability']**(1/df_metrics['amount of data'])
        self.df_metrics = df_metrics.set_index(['symbol', 'tf', 'rule_no.', 'delay', 'side'])
        return self.df_metrics

    def start_backtesting(self, tf=None, delay=0, sensitivity=0, v2=False):
        start = time.time()
        tf = self.tf if not tf else tf
        self.set_timeframe(tf)

        self.df_tf = self.load_timeframe_data()
        # print("Loaded Timeframe data", time.time()-start)
        trade_history = self.calculate_trading_history(delay=delay, sensitivity=sensitivity, v2=v2)
        # print("Calculated Trading History", time.time()-start)
        self.df_th = self.create_trade_history_table(trade_history)
        # print("Trading History Table Created", time.time()-start)
        self.detailed_th = self.generate_detailed_trading_history(self.df_tf, self.df_th).set_index(['tid', 'candle_id'])
        # print("Detailed Th created", time.time()-start)
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

