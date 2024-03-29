import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from doctest import master

import numpy as np
import pandas as pd
import requests
import sqlalchemy
from binance.exceptions import *
from IPython.display import clear_output, display
from playsound import playsound

from backtester import Backtester
from dataloader import _DataLoader
from fab_strategy import FabStrategy
from helper import Helper


class Screener:

    def __init__(self, db=False):
        self.loader = _DataLoader(db=db, ib=False)
        self.master_screener = pd.DataFrame()
        self.strategy = FabStrategy()
        self.tf_delay_match = {235: 0, 145:7, 210:5, 165:4, 130:13, 220:12}
        self.list_of_enters = {
                            ('Rule 1', 'Long'): self.strategy.rule_1_buy_enter,
                            ('Rule 1', 'Short'): self.strategy.rule_1_short_enter,
                            ('Rule 2', 'Long'):  self.strategy.rule_2_buy_enter,
                            ('Rule 2', 'Short'): self.strategy.rule_2_short_enter,
                            ('Rule 3', 'Long'): self.strategy.rule_3_buy_enter,
                            ('Rule 3', 'Short'): self.strategy.rule_3_short_enter
                          }

        self.list_of_exits = {
                            ('Rule 1', 'Long'): self.strategy.rule_1_buy_exit,
                            ('Rule 1', 'Short'): self.strategy.rule_1_short_exit,
                            ('Rule 2', 'Long'):  self.strategy.rule_2_buy_stop_absolute,
                            ('Rule 2', 'Short'): self.strategy.rule_2_short_stop_absolute,
                            ('Rule 3', 'Long'): self.strategy.rule_1_buy_exit,
                            ('Rule 3', 'Short'): self.strategy.rule_1_short_exit
                          }

######################################################################################################################
    async def _async_get_questrade_dfs(self, trader, symbols, tfs, daily_candles=350):
        tf_map = {1: "OneMinute", 5: "FiveMinutes", 15: "FifteenMinutes", 30: "HalfHour", 
                  60: "OneHour", 240: "FourHours", 1440: "OneDay"}
        results = []
        for symbol in symbols:
            for tf in tfs:
                start_datetime, end_datetime = Helper.datetime_from_tf(tf)
                coroutine = trader.loader._async_get_fast_questrade_data(symbol, start_datetime, end_datetime, tf_map[tf], tf)
                results.append(coroutine)
        results = await asyncio.gather(*results, return_exceptions=True)
        return results

    async def _async_get_ibkr_dfs(self, trader, symbols, tfs, daily_candles=350):
        tf_map = {1: "1 min", 5: "5 mins", 15: "15 mins", 30: "30 mins", 60: "1 hour", 240: "4 hours", 1440: "1 day"}  
        count, count2, results = 0, 0, []
        start = time.perf_counter()
        for symbol in symbols:
            for tf in tfs:
                count, count2 = count+1, count2+1 
                if count2 > 5:
                    if time.perf_counter() - start > 10:
                        print(time.perf_counter()-start)
                        print('sleeping')
                        time.sleep(6)
                        count2, start = 0, time.perf_counter()
                    elif time.perf_counter() - start > 5:
                        print(time.perf_counter()-start)
                        print('sleeping')
                        time.sleep(3)
                        count2, start = 0, time.perf_counter()
                if count > 45:
                    time.sleep(1)
                start_datetime, end_datetime = Helper.datetime_from_tf(tf)
                duration = (end_datetime - start_datetime).days + 1
                try:
                    coroutine = await trader.loader._async_get_fast_ibkr_data(symbol, duration, end_datetime, tf_map[tf], tf)
                    results.append(coroutine)
                except Exception as e:
                    results.append(e)
        # results = await asyncio.gather(*results, return_exceptions=True)
        return results

#####################################################################################################################
    
    def _check_for_tf_signals(self, df, enter=True, v2=False, max_candle_history=10):
        
        tf_rule_match = {240: ('Rule 1', 'Long'), 235:('Rule 1', 'Long'),
                        145: ('Rule 1', 'Short'),
                        210: ('Rule 2', 'Long'),
                        130: ('Rule 3', 'Long'), 165: ('Rule 3', 'Long'),
                        220: ('Rule 3', 'Short')}

        tf = df.reset_index()['tf'].iloc[0]
        rule, side = tf_rule_match[tf]
        
        self.strategy.delay = self.tf_delay_match[tf]
        self.sma_df = self.strategy.load_data(df)
                        
        for x_last_row in range(-max_candle_history, 0):
            if enter:
                if self.list_of_enters[(rule, side)](x_last_row) != True:
                    continue
                for remaining_row in range(x_last_row + 1, 0):
                    if self.list_of_exits[(rule, side)](remaining_row) == True:
                        return False, None, None, None, None
                return True, x_last_row, rule, side, self.strategy.delay
            if not enter:
                # print(x_last_row,  list_of_exits[(rule, side)](x_last_row))
                if self.list_of_exits[(rule, side)](x_last_row) == True:
                    return True, x_last_row, rule, side, self.strategy.delay

        return False, None, None, None, None
        

    def _check_for_all_signals(self, df, enter=True, max_candle_history=10, v2=False):
        if v2:
            self.strategy.rule_2_buy_enter = self.strategy.rule_2_buy_enter_v2
            self.strategy.rule_2_short_enter = self.strategy.rule_2_short_enter_v2

        self.strategy.load_data(df)

        for rule, side in self.list_of_enters:
            for x_last_row in range(-max_candle_history, 0):
                if enter:
                    if self.list_of_enters[(rule,side)](x_last_row) != True:
                        continue
                    for remaining_row in range(x_last_row + 1, 0):
                        if self.list_of_exits[(rule, side)](remaining_row) == True:
                            return False, None, None, None, 0
                    return True, x_last_row, rule, side, 0
                if not enter:
                    # print(x_last_row,  list_of_exits[(rule, side)](x_last_row))
                    if self.list_of_exits[(rule, side)](x_last_row) == True:
                        return True, x_last_row, rule, side, 0

        return False, None, None, None, 0

    def _get_questrade_dfs(self, trader, symbols, tfs, daily_candles=250):

        tf_map = {1: "OneMinute", 5: "FiveMinutes", 15: "FifteenMinutes", 30: "HalfHour", 
                  60: "OneHour", 240: "FourHours", 1440: "OneDay"}

        with ThreadPoolExecutor(max_workers=15) as executor:
            results = []
            for symbol in symbols:
                for tf in tfs:
                    start_datetime, end_datetime = Helper.datetime_from_tf(tf, qtrade=True, daily_1m_candles=daily_candles)
                    future = executor.submit(trader.loader._get_fast_questrade_data, symbol, start_datetime, end_datetime, tf_map[tf], tf)
                    results.append((symbol, future))
            executor.shutdown(wait=True)
        return results

    def _clean_futures_results(self, futures_results, length_filter=231):
        clean_results, under_length_results, error_results = [], [], []
        symbol_tf_list = []
        for futures_result in futures_results:
            try:
                result = futures_result[1].result()
                symbol = result.iloc[0,0]
                tf = result.iloc[0,1]
                length = len(result)
                if length > length_filter:
                    symbol_tf_list.append([symbol,tf])
                    clean_results.append(futures_result[1].result())
                else:
                    under_length_results.append(futures_result.result())
            except:
                error_results.append(futures_result)
                continue

        self.under_length_results = under_length_results
        self.error_results = error_results
        symbol_tf_df = pd.DataFrame(symbol_tf_list, columns = ['symbols', 'tf'])

        return clean_results, symbol_tf_df
        
    def _assemble_master_stock_screener(self, df_results, symbol_tf_df, finviz_df=None, reddit_df=None):
        signal = None
        master_screener = pd.DataFrame(columns=['symbol', 'tf', 'date', 'rule', 'side', 'how recent', '% change'])

        for df, (symbol, tf) in zip(df_results, symbol_tf_df.values):
            try:
                signal, x_many_candles_ago, rule, side, delay = self._check_for_all_signals(df, max_candle_history=10, v2=True, enter=True)
            except:
                print(f"{symbol, tf} didn't work")
            
            if signal:
                symbol = df['symbol'].iloc[0]
                tf = df['tf'].iloc[0]
                date = datetime.now() - timedelta(minutes=int(abs(x_many_candles_ago)*tf))
                change_factor = float(df['close'].iloc[-1]/df['close'].iloc[x_many_candles_ago])
                percentage_change = Helper.factor_to_percentage([change_factor])[0]
                master_screener = master_screener.append(pd.DataFrame([[symbol, tf, date, rule, side, x_many_candles_ago, percentage_change]],
                                                    columns=['symbol', 'tf', 'date', 'rule', 'side', 'how recent', "% change"]))

        master_screener['most recent'] = master_screener.groupby('symbol')['how recent'].transform('max')

        if type(finviz_df) != type(None):
            finviz_df = finviz_df.rename(columns={'Ticker':'symbol'})
            finviz_df = finviz_df[['symbol', 'Market Cap', 'Price', 'Volume']]
            master_screener = finviz_df.merge(master_screener, on='symbol')

        if type(reddit_df) != type(None):
            reddit_df = reddit_df.rename(columns={'ticker':'symbol'})
            master_screener = master_screener.merge(reddit_df, on='symbol', how='left')
       
        return  master_screener.set_index(['symbol', 'tf']).sort_values(['most recent', 'how recent'], ascending=[False, False])
    
    def _load_reddit_mentions(self, crypto=True):
        url = f"https://apewisdom.io/api/v1.0/filter/all-crypto/page/"
        if not crypto:
            url = f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/" 
        
        pages = requests.get(url + "0").json()['pages']
        
        df = pd.DataFrame()
        
        for page in range(1,pages):
            js = requests.get(url + str(page)).json()
            df = df.append(pd.json_normalize(js['results']))
        return df.drop(['rank', 'rank_24h_ago'], axis=1)

    def stock_screen(self, trader, finviz_df, tfs, reddit_df=None):
        symbols = finviz_df['Ticker'].values
        self.futures_results = self._get_questrade_dfs(trader, symbols, tfs)
        self.clean_results, self.symbol_tf_df  = self._clean_futures_results(self.futures_results, length_filter=231)
        master_screener = self._assemble_master_stock_screener(df_results=self.clean_results, symbol_tf_df=self.symbol_tf_df, finviz_df=finviz_df, reddit_df=reddit_df)
        return master_screener
    

    def _get_binance_dfs(self, trader, symbols, max_candles_needed=245):
        with ThreadPoolExecutor(max_workers=9) as executor:
            results = [(symbol, executor.submit(trader.set_asset, symbol, 5, max_candles_needed*48, False)) for symbol in symbols['symbol'].values]
            executor.shutdown(wait=True)
        return results

    def _assemble_screener_without_metrics(self, partial_screener, df, x_many_candles_ago, tf, rule, side):
        symbol = df['symbol'].iloc[0]
        date = datetime.now() - timedelta(minutes=int(abs(x_many_candles_ago)*tf))
        change_factor = float(df['close'].iloc[-1]/df['close'].iloc[x_many_candles_ago])
        percentage_change = Helper.factor_to_percentage([change_factor])[0]
        partial_screener = partial_screener.append(pd.DataFrame([[symbol, tf, date, rule, side, x_many_candles_ago, percentage_change]],
                                            columns=['symbol', 'tf', 'date', 'rule', 'side', 'how recent', "% change"]))
        return partial_screener
    def _assemble_master_crypto_screener(self, trader, metrics_table, symbols, tfs, results, v2=None, skip_new_data=True, tf_only=True, show_all_trades=False):
        partial_screener = pd.DataFrame()
        metrics_bool = True
        self.df_dic = {}

        if len(results) == 0:
            return pd.DataFrame()
    
        for df, symbol in zip(results, symbols['symbol'].values):
            for tf in tfs:
                df_tf = trader.loader._timeframe_setter(df, tf//5, keep_last_row=False)
                self.df_dic[symbol,tf] = df_tf

                if tf_only:
                    signal, x_many_candles_ago, rule, side, delay = self._check_for_tf_signals(df_tf, max_candle_history=10, v2=v2, enter=True)
                else:
                    signal, x_many_candles_ago, rule, side, delay = self._check_for_all_signals(df_tf, max_candle_history=10, v2=v2, enter=True)
                
                if signal:
                    try:
                        metric_id = int(metrics_table.loc[(symbol, tf, rule, side, delay), 'metric id'])
                    except Exception:
                        if skip_new_data and tf_only:
                            continue
                        metrics_bool = False
                        partial_screener = self._assemble_screener_without_metrics(partial_screener, df, x_many_candles_ago, tf, rule, side)
                        continue
                    date = datetime.now() - timedelta(minutes=int(abs(x_many_candles_ago)*tf))
                    change_factor = float(df['close'].iloc[-1]/df['close'].iloc[x_many_candles_ago])
                    percentage_change = Helper.factor_to_percentage([change_factor])[0]
                    partial_screener = partial_screener.append(pd.DataFrame([[metric_id, symbol, tf, date, rule, side, x_many_candles_ago, percentage_change]],
                                                        columns=['metric id', 'symbol', 'tf', 'date', 'rule', 'side', 'how recent', "% change"]))
        if len(partial_screener) == 0:
            return pd.DataFrame(columns=['metric id', 'symbol', 'tf', 'date', 'rule', 'side', 'how recent', "% change"])

        if metrics_bool:
            master_screener = partial_screener.merge(metrics_table.reset_index()
                                                      [['metric id', 'amount of data', 'win loss rate', 'profitability', 
                                                      'avg pnl', 'longest drawdown','longest run', 'num candles to peak']],
                                                      on='metric id')
            master_screener['most recent'] = master_screener.groupby('symbol')['how recent'].transform('max')  
            delta = pd.to_timedelta(master_screener['num candles to peak']*master_screener['tf'], 'minutes')
            master_screener['due date'] = master_screener['date'] + delta
            master_screener = master_screener.set_index(['symbol', 'tf']).sort_values(
                ['most recent', 'how recent', 'avg pnl'], ascending=[False, False, False])
            master_screener = master_screener[['metric id'] + list(master_screener.columns.drop('metric id'))]
            if show_all_trades:
                return master_screener
            master_screener = master_screener[master_screener['amount of data'] >= 5]
            master_screener = master_screener[master_screener['avg pnl']>1.02]
        else:
            master_screener = partial_screener
            master_screener['most recent'] = master_screener.groupby('symbol')['how recent'].transform('max')  
            master_screener = master_screener.set_index(['symbol', 'tf']).sort_values(
                ['most recent', 'how recent'], ascending=[False, False])
        return master_screener
    def _check_futures_rate_limit_error(self, error_results):
        if len(error_results) > 0:
            for symbol, error in error_results:
                try:
                    error.result()
                except BinanceAPIException as e:
                    raise(e)

    def crypto_screen(self, trader, metrics_table, tfs, max_candle_history=10, max_candles_needed=245, v2=False, tf_only=True):
        # First line filters all symbols that historically didn't perform
        symbols = pd.DataFrame(metrics_table[metrics_table['profitability'] > 1].index.unique(level=0))
        max_candles_needed = max_candles_needed + max_candle_history + 1
        self.results = self._get_binance_dfs(trader, symbols=symbols, max_candles_needed=max_candles_needed)
        self.clean_results, symbol_tf_df  = self._clean_futures_results(self.results, length_filter=231)
        self._check_futures_rate_limit_error(self.error_results)
        master_screener = self._assemble_master_crypto_screener(trader, metrics_table, symbols, tfs, self.clean_results, tf_only=tf_only)
        return master_screener

    def crypto_screen_parts(self, trader, metrics_table, tfs, max_candle_history=10, max_candles_needed=245, v2=False, tf_only=True):
        # First line filters all symbols that historically didn't perform
        symbols = pd.DataFrame(metrics_table[metrics_table['profitability'] > 1].index.unique(level=0))
        symbols_1 = symbols[:35]
        symbols_2 = symbols[35:70]
        symbols_3 = symbols[70:105]
        max_candles_needed = max_candles_needed + max_candle_history + 1
        results_1 = self._get_binance_dfs(trader, symbols=symbols_1, max_candles_needed=max_candles_needed)
        # time.sleep(60)
        results_2 = self._get_binance_dfs(trader, symbols=symbols_2, max_candles_needed=max_candles_needed)
        # time.sleep(60) 
        results_3 = self._get_binance_dfs(trader, symbols=symbols_3, max_candles_needed=max_candles_needed)
        results = results_1 + results_2 + results_3
        self.results = results
        self.clean_results, symbol_tf_df  = self._clean_futures_results(results, length_filter=231)

        self._check_futures_rate_limit_error(self.error_results)

        master_screener = self._assemble_master_crypto_screener(trader, metrics_table, symbols, tfs, self.clean_results)
        return master_screener

    def get_metrics_table(self, v2=True):
        cursor = self.loader.sql.conn.cursor()
        if v2:
            df_metrics = self.loader.sql.SELECT(f"* from metrics_v2 order by symbol", cursor)
            df_metrics = Helper.remove_column_underscore(df_metrics).set_index(['symbol', 'tf', 'rule no', 'side', 'delay']).astype(float)
            df_metrics[['metric id', 'amount of data', 'num candles to peak']] = df_metrics[['metric id', 'amount of data', 'num candles to peak']].astype(int)
        else:
            df_metrics = self.loader.sql.SELECT(f"* from metrics order by symbol", cursor).set_index(
                ['symbol', 'tf', 'rule_no', 'side']).astype(float)
        return df_metrics

    def screener_monitor(self, trader, minutes_to_sleep=1):
        df_metrics = self.get_metrics_table()
        count = 0

        now = datetime.now()
        now = datetime(year=now.year, month=now.month, day=now.day, hour=now.hour, minute=now.minute)
        while True:
            if datetime.now() >= now + pd.Timedelta(1, 'minute'):
                count += 1
                print(datetime.now())
                now = datetime.now().minute
                master_screener = self.screen(trader=trader, metrics_table=df_metrics)
                clear_output(wait=True)
                if count > 1 and len(master_screener) != len(self.master_screener):
                    new_values = []
                    temp = pd.concat([master_screener, self.master_screener]).drop_duplicates(subset='metric_id', keep=False)
                    for symbol, tf in temp.index:
                        try:
                            if temp.loc[(symbol, tf), 'how recent'] == -1:
                                new_values.append((symbol, tf))
                        except Exception():
                            clear_output(wait=True)
                            print(temp, symbol, tf)
                    if len(new_values) > 0:
                        playsound(r"C:\Users\haseab\Desktop\Python\PycharmProjects\FAB\local\cash_register.mp3")
                        print("new values added: ")
                        print(new_values)

                self.master_screener = master_screener.copy()

                display(self.master_screener[self.master_screener['amount_of_data'] >= 5].head(60))
                time.sleep(3)
                time.sleep(10 * minutes_to_sleep)
                time.sleep(10 * minutes_to_sleep)
                time.sleep(10 * minutes_to_sleep)
                time.sleep(10 * minutes_to_sleep)
                time.sleep(10 * minutes_to_sleep)

            time.sleep(0.5)


    def _add_signals_to_list(self, recency, top_n_trades):
        top_recent_trades = []
        for trade in top_n_trades.index:
            if top_n_trades.loc[trade, 'how recent'] == recency:
                side = top_n_trades.loc[trade, 'side']
                rule = top_n_trades.loc[trade, 'rule']
                symbol = trade[0]
                top_recent_trades.append((symbol, side, rule))
        return top_recent_trades

    def _update_recent_signals_csv(self, top_n_trades):
        old_signals = pd.read_csv("Recent signals.txt").set_index(['symbol','tf'])
        recent_signals = old_signals.append(top_n_trades).reset_index().drop_duplicates(subset='symbol', keep='last')
        recent_signals.to_csv("Recent signals.txt", index=False)

    def top_trades(self, trader, df_metrics, tfs, n=3, full=False, recency=-1, top_n_trades=()):
        print("Getting Top New Trades:.... ")
        if len(top_n_trades) ==0:
            top_n_trades = self.crypto_screen_parts(trader=trader, metrics_table=df_metrics, tfs=tfs)
            self._update_recent_signals_csv(top_n_trades)
        df_filtered = top_n_trades[top_n_trades['how recent'] == recency].head(n)
        if full:
            df_filtered = top_n_trades
        if len(df_filtered) == 0:
            print("Found no new trades at this time", end='\n\n')
        top_recent_trades = self._add_signals_to_list(recency, top_n_trades)
        return top_recent_trades, df_filtered






