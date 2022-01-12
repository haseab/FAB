import math
import os
import random
import time
from datetime import datetime
from hmac import new

import dateparser
import numpy as np
import pandas as pd
from binance.client import Client
from decouple import config
# from ftx import FtxClient
from IPython.display import clear_output, display

from dataloader import _DataLoader
from fab_strategy import FabStrategy
from helper import Helper
from illustrator import Illustrator
from screener import Screener
from trade_executor import TradeExecutor
from trading_history import TradeHistory


class Trader():
    """
    Purpose is to trade live on the Binance Crypto Exchange

    Attributes
    -----------
    symbol: str
    client: Client object - From binance.py
    capital: float - represents how much money is in account
    leverage: float - how many multiples of your money you want to trade
    tf: int - timeframe
    df: pd.DataFrame - data that will be analyzed.
    Methods
    ----------
    load_account
    set_leverage
    set_timeframe
    get_position
    get_necessary_data
    set_asset
    make_row
    load_existing_asset
    start_trading

    Please look at each method for descriptions
    """
    max_candle_history = 231

    def __init__(self, binance=True, qtrade=False, db=False, ib=False):
        if binance:

            self.binance = Client()
        # self.ftx = FtxClient()
        self.capital = None
        self.leverage = 1 / 1000
        self.tf = None
        self.executor = TradeExecutor()
        self.symbol = None
        self.trade_metrics = None
        self.loader = _DataLoader(db=db, qtrade=qtrade, ib=ib)
        self.illustrator = Illustrator()
        self.strategy = FabStrategy()
        self.df = None
        self.order_history = pd.DataFrame()
        self.trade_replay = []
        self.precisions = {symbol_dic['symbol'] : symbol_dic['quantityPrecision'] for symbol_dic in self.loader.binance.futures_exchange_info()['symbols']}

    def check_on_switch(self) -> bool:
        """
        Reads a txt file to see whether user wants trading to be on or off
        :return boolean of True or False. True means trading is on, and false means it is off.
        """
        local_path = os.path.dirname(os.path.dirname(os.path.abspath("__file__"))) + r"\local\trade_status.txt"
        status = pd.read_csv(local_path).set_index('index')
        self.start = bool(status.loc[0, 'status'])
        return self.start

    def load_account(self, additional_balance: int = 0) -> str:
        """
        Sign in to account using API_KEY and using Binance API
        """
        # local_path_binance = os.path.dirname(os.path.dirname(os.path.abspath("__file__"))) + r"\local\binance_api.txt"
        # local_path_ftx = os.path.dirname(os.path.dirname(os.path.abspath("__file__"))) + r"\local\ftx_api.txt"
        # info_binance = pd.read_csv(local_path_binance).set_index('Name')
        # info_ftx = pd.read_csv(local_path_ftx).set_index('Name')

        API_KEY_BINANCE = config('API_KEY_BINANCE')
        API_SECRET_BINANCE  = config('API_SECRET_BINANCE')

        # API_KEY_FTX = config('API_KEY_FTX')
        # API_SECRET_FTX  = config('API_SECRET_FTX')

        self.binance = Client(api_key=API_KEY_BINANCE, api_secret=API_SECRET_BINANCE)
        # self.ftx = FtxClient(api_key=API_KEY_FTX, api_secret=API_SECRET_FTX)
        return "Connected to Binance"

    def get_capital(self, additional_balance=0, binance_only=False, ):

        asset_prices = {symbol['symbol'][:-4]: float(symbol['lastPrice']) for symbol in self.binance.get_ticker() if symbol['symbol'][-4:] == 'USDT'}

        # Binance  account balance
        binance_account = self.binance.futures_account()
        binance_futures_balance = float(binance_account['totalCrossWalletBalance']) # + float(binance_account['totalCrossUnPnl'])
        binance_asset_balances = {symbol['asset']: (float(symbol['free']) + float(symbol['locked'])) for symbol in self.binance.get_account()['balances']}
        binance_usdt_balance = binance_asset_balances["USDT"]

        binance_spot_balance = 0
        for binance_asset in binance_asset_balances:
            if binance_asset in asset_prices:
                binance_spot_balance += asset_prices[binance_asset] * binance_asset_balances[binance_asset]

        binance_balance = binance_spot_balance + binance_usdt_balance + binance_futures_balance

        if binance_only:
            self.capital = binance_balance + additional_balance
            return self.capital
        # FTX  account balanced
        ftx_account = self.ftx.get_account_info()
        ftx_balance = ftx_account['totalAccountValue']

        # ftx_asset_balances = {symbol['coin']:symbol['total'] for symbol in self.ftx.get_balances()}
        # ftx_usdt_balance = ftx_asset_balances['USDT'] + ftx_asset_balances['USD']
        #
        # ftx_spot_balance = 0
        # for ftx_asset in ftx_asset_balances:
        #     if ftx_asset in asset_prices:
        #         ftx_spot_balance += asset_prices[ftx_asset] * ftx_asset_balances[ftx_asset]
        # ftx_balance = ftx_spot_balance + ftx_usdt_balance

        self.capital = ftx_balance + binance_balance + additional_balance
        return self.capital


    def load_futures_trading_history(self, start_time, end_time=None):
        df = pd.DataFrame()
        ranges = Helper.determine_timestamp_positions(start_time, end_time, limit=7) 

        for index, timestamp in enumerate(ranges):
            try:
                df = df.append(pd.DataFrame(self.binance.futures_account_trades(startTime=ranges[index], endTime=ranges[index+1])))
            except (IndexError) as e:
                pass
        return df.reset_index(drop=True)

    def set_leverage(self, leverage: float) -> float:
        """Sets the current leverage of the account: should normally be 1. And 0.001 for testing purposes"""
        self.leverage = leverage
        return self.leverage

    def set_timeframe(self, tf: int) -> int:
        """Sets the timeframe of the trading data"""
        self.tf = tf
        return self.tf

    def show_live_chart(self, symbol, tf, refresh_rate=1):
        while True:
            time.sleep(refresh_rate)
            clear_output(wait=True)
            self.show_current_chart('BTCUSDT', tf)

    def show_current_chart(self, symbol=None, tf=None, metric_id=None):
        if metric_id:
            cursor = self.loader.conn.cursor()
            df_metrics = self.loader.sql.SELECT(f"* FROM metrics where metric_id = {metric_id}", cursor)[['symbol', 'tf']]
            symbol = df_metrics['symbol'].iloc[0]
            tf = df_metrics['tf'].iloc[0]

        data = self.set_asset(symbol, tf, max_candles_needed=375, drop_last_row=False)
        return self.illustrator.graph_df(data)

    def get_necessary_data_v2(self, symbol: str, tf: int, max_candles_needed:int = 235) -> pd.DataFrame:
        now = time.time()
        df = pd.DataFrame()

        # days = max_candles_needed*tf//1440 + 1
        start_datetime = str(Helper.datetime_from_tf(tf, daily_1m_candles=1440, max_candles_needed=max_candles_needed)[0])
        df = df.append(self.loader.get_binance_candles(symbol, tf, start_date=start_datetime))
        df['symbol'] = [symbol for _ in range(len(df))]
        return df.drop_duplicates()


    def get_necessary_data(self, symbol: str, tf: int, max_candles_needed:int = 245) -> pd.DataFrame:
        """
        Gets the minimum necessary data to trade this asset.

        Note: This method is used as a way to tackle the 1000 candle limit that is currently on the Binance API.
        A discrete set of ~1000 group candles will be determined, and then the data will be extracted from each,
        using the _get_binance_futures_candles method, and then all of them will be merged together.

        Parameters:
        symbol: str             - Symbol of price ticker   Ex. "BTCUSDT", "ETHUSDT"
        tf: int                 - Timeframe wanted   Ex. 1, 3, 5, 77, 100
        max_candles_needed: int - Maximum candles needed in the desired timeframe     Ex. 231, 770, 1440

        :return pd.DataFrame of candlestick data

        """
        now = time.time()
        df = pd.DataFrame()

        crypto = 'USDT' in symbol or 'BUSD' in symbol

        if crypto:
            ranges = Helper.determine_candle_positions(max_candles_needed, tf) 
            for i in range(len(ranges)):
                try:
                    df = df.append(self.loader._get_binance_futures_candles(symbol, tf, int(ranges[i]), int(ranges[i + 1]), now))
                except IndexError:
                    pass
            df['symbol'] = [symbol for _ in range(len(df))]
            return df.drop_duplicates()
        else:
            tf_map = {1: "OneMinute", 5: "FiveMinutes", 15: "FifteenMinutes", 30: "HalfHour", 
                  60: "OneHour", 240: "FourHours", 1440: "OneDay"}
            daily_candles = 350
            how_many_days_ago = max_candles_needed*tf//daily_candles + 1
            start_time, end_time = f"{how_many_days_ago} days ago" , 'now'
            parsed_start, parsed_end = dateparser.parse(start_time), dateparser.parse(end_time)
            parsed_start, parsed_end = parsed_start.strftime('%Y-%m-%d %H:%M:%S.%f'), parsed_end.strftime('%Y-%m-%d %H:%M:%S.%f')
            data = self.loader.qtrade.get_historical_data(symbol, parsed_start, parsed_end, tf_map[tf])
            return Helper.into_dataframe(data, symbol=symbol, tf=tf)

    def _update_data(self, diff: int, symbol: str, ) -> None:
        """
        Used to update the trading data with the exchange data so that it is real time

        Parameters:
        -----------
        diff: a number that explains how many minutes of disparity there is between current data and live data.

        :return dataframe with the most up-to-date data.
        """
        minutes_disparity = math.floor(diff) + 1

        # Getting minute candlestick data. Number of minute candlesticks represented by "minutes" variable
        last_few_candles = self.loader._get_binance_futures_candles(symbol, minutes_disparity)
        # Adding a legible Datetime column and using the timestamp data to obtain datetime data
        last_few_candles['datetime'] = Helper.millisecond_timestamp_to_datetime(last_few_candles.index)
        last_few_candles = last_few_candles[["datetime", "open", "high", "low", "close", "volume"]]

        # Abstracting minute data into appropriate tf
        last_few_candles = self.loader._timeframe_setter(last_few_candles, self.tf, 0)

        # Updating original data with new data.
        self.df = self.df.append(last_few_candles).drop_duplicates()

    def get_positions_amount(self, list_of_symbols: list, divide_by=None, max_trades=3) -> list:
        """
        Gets the total amount of current position for a given symbol
        Parameters:
        ------------
        symbol: str
            Ex. "BTCUSDT"
                "ETHUSDT"

        Returns float
            Ex. If a 1.000 BTC position is open, it will return 1.0
        """
        position_values = {}
        for symbol in list_of_symbols:
            last_price = float(self.binance.futures_mark_price(symbol=symbol)['markPrice'])
            for coin_futures_info in self.binance.futures_position_information():
                if coin_futures_info['symbol'] == symbol:
                    amount = float(coin_futures_info["positionAmt"])
                    amount = round(amount, self.precisions[symbol])
                    if divide_by:
                        amount = round(amount - amount/divide_by, self.precisions[symbol])
                        print(amount*last_price)
                        if self.capital / max_trades > amount*last_price:
                            return None
                        position_values[symbol] = amount if amount > 0 else 10**(-self.precisions[symbol])
                    else:
                        position_values[symbol] = amount 
        return position_values

    def get_single_asset_signals(self, screener, symbol, tf, enter=True):
        df = self.set_asset_v2(symbol, tf, max_candles_needed=245)
        binance_df = int(df['tf'].iloc[0])
        df_tf = self.loader._timeframe_setter(df, tf//binance_df, keep_last_row=True, shift=0)
        strat = FabStrategy()
        df_ma = strat.load_data(df_tf)
        return df_ma, screener._check_for_tf_signals(df_ma, max_candle_history=10, enter=enter)
        
    def set_asset_v2(self, symbol: str, tf: int, max_candles_needed: int = 245, v2=False) -> pd.DataFrame:
        binance_tf = Helper.find_greatest_divisible_timeframe(tf)
        new_max_candles = max_candles_needed*(tf//binance_tf)
        if v2:
            df = self.get_necessary_data_v2(symbol, binance_tf, new_max_candles)
        else:
            df = self.get_necessary_data(symbol, binance_tf, new_max_candles)
        self.df, self.tf, self.symbol= df, tf, symbol
        return df

    def set_asset(self, symbol: str, tf: int, max_candles_needed: int = 245, keep_last_row=False) -> pd.DataFrame:
        """
        Set Symbol of the symbol and load the necessary data with the given timeframe to trade it

        Parameters:
        ------------
        symbol: str     Ex. "BTCUSDT", "ETHUSDT"

        :return str response
        """
        # For Binance API purposes, 240 min needs to be inputted as "4h" on binance when fetching data
        map_tf = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h", 360: "6h", 480: "8h"}
        minutes_ago = tf * (max_candles_needed + 3)
        start_time = Helper().minutes_ago_to_timestamp(minutes_ago, time.time(), 1000)

        if tf in map_tf.keys():  # Note: 12H, 1D, 3D, 1W, 1M are also recognized
            # Fetching data from Binance if it matches the eligible timeframe, as it will be faster
            df = Helper.into_dataframe(
                self.binance.futures_klines(symbol=symbol, interval=map_tf[tf], startTime=start_time), symbol=symbol, tf=tf)
            df = df.reset_index().set_index(['symbol', 'tf', 'timestamp'])
            if not keep_last_row:
                df.drop(df.tail(1).index, inplace=True)
        else:
            # If it doesn't match Binance available timeframes, it must be transformed after fetching the least divisible timeframe data.
            binance_tf = Helper.find_greatest_divisible_timeframe(tf)
            df = self.get_necessary_data(symbol, binance_tf, max_candles_needed*(tf//binance_tf))
            df = self.loader._timeframe_setter(df, tf//binance_tf, keep_last_row=keep_last_row)

        # Adding Datetime column for readability of the timestamp. Also more formatting done
        df = df[["date", "open", "high", "low", "close", "volume"]]

        self.df = df
        self.tf = tf
        self.symbol = symbol
        return df

    def get_current_tf_candle(self, tf):
        minute_candles = self.loader._get_binance_futures_candles("BTCUSDT", tf)
        minute_candles['Datetime'] = Helper.millisecond_timestamp_to_datetime(minute_candles.index)
        minute_candles = minute_candles[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

        tf_candle = self.loader._timeframe_setter(minute_candles, tf, keep_last_row=True)
        return tf_candle

    def load_existing_asset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sets the trading data to an already existing dataframe, passed in as an argument"""
        self.df = df
        return self.df

    def check_rule_1(self, client, strategy, executor, position_amount):
        if strategy.rule_1_buy_enter(-1) and position_amount == 0:
            trade_info = executor.enter_market(client, self.symbol, "BUY", self.capital, self.leverage, 1)
        elif strategy.rule_1_buy_exit(-1) and position_amount > 0:
            trade_info = executor.exit_market(client, self.symbol, 1, position_amount)
        elif strategy.rule_1_short_enter(-1) and position_amount == 0:
            trade_info = executor.enter_market(client, self.symbol, "SELL", self.capital, self.leverage, 1)
        elif strategy.rule_1_short_exit(-1) and position_amount < 0:
            trade_info = executor.exit_market(client, self.symbol, 1, position_amount)

    def check_rule_2(self, client, strategy, executor, sensitivity, position_amount):
        if strategy.rule_2_buy_enter(-1, sensitivity) and position_amount == 0:
            trade_info = executor.enter_market(client, self.symbol, "BUY", self.capital, self.leverage, 2)
        elif strategy.rule_2_short_enter(-1, sensitivity) and position_amount == 0:
            trade_info = executor.enter_market(client, self.symbol, "SELL", self.capital, self.leverage, 2)
        elif strategy.rule_2_short_stop(-1) and position_amount < 0 and \
                executor.live_trade_history.last_trade().rule == 2:
            trade_info = executor.exit_market(client, self.symbol, 2, position_amount)
        elif strategy.rule_2_buy_stop(-1) and position_amount > 0 and \
                executor.live_trade_history.last_trade().rule == 2:
            trade_info = executor.exit_market(client, self.symbol, 2, position_amount)

    def check_rule_3(self, client, strategy, executor, position_amount):
        if strategy.rule_3_buy_enter(-1) and position_amount == 0:
            trade_info = executor.enter_market(client, self.symbol, "BUY", self.capital, self.leverage, 3)
        elif strategy.rule_1_buy_exit(-1) and position_amount > 0:
            trade_info = executor.exit_market(client, self.symbol, 1, position_amount)
        elif strategy.rule_3_short_enter(-1) and position_amount == 0:
            trade_info = executor.enter_market(client, self.symbol, "SELL", self.capital, self.leverage, 3)
        elif strategy.rule_1_short_exit(-1) and position_amount < 0:
            trade_info = executor.exit_market(client, self.symbol, 1, position_amount)

    def start_trading(self, strategy: FabStrategy, executor, tf: int,  sensitivity=0, debug=False) -> str:
        """
        Starts the process of trading live. Each minute, the last row of the data is updated and Rule #2 is checked,
        otherwise, it waits till the end of the timeframe where it gets the latest data before it checks the rest of the rules.
        If it does see something following the rules, it will buy/short, given the initial parameters it has. (e.g. leverage, quantity)
        This process continues indefinetly, unless interrupted.

        Returns None.
        """

        raise Exception("Need to review method before running ")

        client = self.binance
        strategy.load_data(self.df)
        strategy.update_moving_averages()

        self.check_on_switch()

        while self.start != False:
            # getting minute candle disparity from real time data
            minute_disparity = Helper.calculate_minute_disparity(self.df, tf)

            if Helper.change_in_clock_minute() and minute_disparity <= tf:
                df_row = self.get_current_tf_candle(minute_disparity)
                position_amount = self.get_position_amount(self.symbol)
                strategy.load_data(self.df.append(df_row))
                strategy.update_moving_averages()

                # Checking only Rule 2, because it needs a minute by minute check.
                self.check_rule_2(client, strategy, executor, sensitivity, position_amount)

                time.sleep(50)

            elif minute_disparity > tf:
                # Updating data using Binance API instead of appending the completed final row from the above code
                self._update_data(math.floor(minute_disparity), self.symbol)
                position_amount = self.get_position_amount(self.symbol)
                strategy.load_data(self.df)
                strategy.update_moving_averages()

                self.check_rule_1(client, strategy, executor, position_amount)
                self.check_rule_3(client, strategy, executor, position_amount)

                time.sleep(1)

            self.check_on_switch()

        return "Trading Stopped"

    def get_current_trade_progress(self, printout=True) -> pd.DataFrame:
        """
        Returns something like

            symbol  side  enter_price    size  usd size  current_price   pnl (%)   pnl (USD)
        0   ETHUSDT  SELL   3981.93000   0.002      7.96    3882.784542  1.024386   0.19416
        1  CELRUSDT   BUY      0.08166  73.000      5.96       0.072870  0.891912   -0.64420
        2   XRPUSDT   BUY      0.81420   7.300      5.94       0.798600  0.980350   -0.11677

        """
        print("Checking Current Trade Progress.... ")
        positions = self.binance.futures_position_information()
        pnl = 1
        trade_progress = pd.DataFrame()
        for symbol_info in positions:
            position_size = float(symbol_info['positionAmt'])
            if position_size == 0:
                continue
            symbol = symbol_info['symbol']
            side = 'BUY' if position_size > 0 else 'SELL'
            enter_price = float(symbol_info['entryPrice'])
            last_price = float(symbol_info['markPrice'])
            if side == 'SELL':
                pnl = Helper.calculate_short_profitability(enter_price, last_price, 0.99975, percentage=True)
            elif side == 'BUY':
                pnl = Helper.calculate_long_profitability(enter_price, last_price, 0.99975, percentage=True)
            usd_size = abs(round(position_size * enter_price, 2))
            pnl_usd = usd_size*(pnl/100)
            trade_progress = trade_progress.append(pd.DataFrame([[symbol, side, enter_price, abs(position_size), usd_size, last_price, pnl, pnl_usd]],
                                                            columns = ['symbol', 'side', 'enter_price', 'size', 'usd size', 'current_price', 'pnl (%)' , 'pnl (USD)']),
                                                ignore_index=True)
            trade_progress['position share'] = trade_progress['usd size'] / trade_progress['usd size'].sum()
            portfolio_pnl = (trade_progress['position share']*trade_progress['pnl (%)']).sum()
        if printout:
            print(f"Portfolio Profit: {round(portfolio_pnl,2)}%")
            display(trade_progress)

        self.trade_progress = trade_progress
        return trade_progress

    def _check_trade_close(self, screener: Screener, current_positions, tfs):
        trades_to_close = []
        current_position_symbols = current_positions['symbol'].values
        # current_position_dfs = {symbol:screener.clean_results[symbol] for symbol in current_position_symbols}

        for symbol in current_position_symbols:
            for tf in tfs:
                try:
                    df = screener.df_dic[(symbol, tf)]
                    enter_signal, enter_x_last_row, *rest= screener._check_for_tf_signals(df, max_candle_history=10, enter=True)
                    close_signal, exit_x_last_row, *rest= screener._check_for_tf_signals(df, max_candle_history=10, enter=False)
                    
                    if close_signal:
                        if enter_x_last_row and exit_x_last_row < enter_x_last_row:
                            continue
                        print(enter_x_last_row, exit_x_last_row)
                        trades_to_close.append(symbol)
                        print(f'{symbol} Trade Closed')
                except AttributeError as e:
                    return []
                
        return trades_to_close

    def output_loading(self):
        print('waiting for next minute...', end='\r')
        time.sleep(0.3)
        print('waiting for next minute.. ', end='\r')
        time.sleep(0.3)
        print('waiting for next minute.  ', end='\r')
        time.sleep(0.3)
        print('waiting for next minute   ', end='\r')
        time.sleep(0.3)
        print('waiting for next minute.  ', end='\r')
        time.sleep(0.3)
        print('waiting for next minute..  ', end='\r')
        time.sleep(0.3)

    def trade_free_capital(self, executor, leverage, remaining_to_invest, trades_to_enter):
        print("-----------------------------------------------")
        print(f"Free capital: {remaining_to_invest} USD")
        print("-----------------------------------------------", end="\n\n\n")

        self.order_history = self.order_history.append(executor.enter_market(client=self.binance, symbol_side_rule=trades_to_enter, capital=remaining_to_invest, 
                                                                             number_of_trades=len(trades_to_enter), reason='free_capital'))
        display(self.order_history.tail(len(trades_to_enter)))
        self.order_history.to_csv("FAB Order History.csv")
        self.trade_replay.append(self.get_current_trade_progress(printout=False))

    def calculate_remaining_capital(self, current_positions, capital, leverage):
        total_invested = current_positions['usd size'].sum() if len(current_positions) > 0 else 0
        remaining_to_invest = capital*leverage - total_invested
        return remaining_to_invest

    def _profit_optimization(self, essential_metrics, current_positions, trades_to_enter, number_of_trades):
        """
        Returns:
            final_trades_to_enter: Ex. [('ZILUSDT', "Short", 'Rule 2'), ('BTCUSDT', 'Long', 'Rule 1'), ("ETHUSDT", "Short", 'Rule 3')]
            trades_to_close:       Ex. ['ADAUSDT']
        """
        print()
        print("Optimizing Trades:... ")
        
        temp_metrics = essential_metrics.reset_index().set_index(['symbol', 'rule no', 'side'])
        trades_of_interest = [(symbol, int(temp_metrics.loc[(symbol, rule, side), 'tf']))  for symbol, side, rule in trades_to_enter]

        ranks = essential_metrics[essential_metrics['amount of data'] >= 5].sort_values('avg pnl', ascending=False)
        pure_ranks = ranks[ranks['avg pnl'] > 1.01].reset_index().set_index(['symbol', 'tf'])
        # key = ranks.groupby('symbol').max().sort_values('avg pnl', ascending=False)
        self.key = pure_ranks

        current_positions_df = pd.read_csv("Recent Signals.txt").set_index(['symbol', 'tf'])
        current_position_symbols_tf_pair = current_positions_df.reset_index().merge(current_positions, on ='symbol', how='inner')[['symbol','tf']].values
        current_position_symbols_tf_pair = Helper().nparray_to_tuple(current_position_symbols_tf_pair)

        all_symbols = list(set(trades_of_interest).union(set(current_position_symbols_tf_pair)))

        self.top_x_symbols = sorted(all_symbols, key=lambda x: self.key.loc[x]['avg pnl'], reverse=True)[:number_of_trades]

        partial_trades_to_enter = list(set(self.top_x_symbols).intersection(set(trades_of_interest)).difference(set(current_position_symbols_tf_pair)))
        trades_to_close = list(set(current_position_symbols_tf_pair).difference(set(self.top_x_symbols)))

        final_trades_to_enter = [(symbol, side, rule) for (symbol, tf), (symbol, side, rule) in zip(trades_of_interest, trades_to_enter) if (symbol,tf) in partial_trades_to_enter]

        return trades_to_close, final_trades_to_enter

    def optimize_trades(self, executor, current_positions, leverage, essential_metrics, trades_to_enter, number_of_trades):
        if len(current_positions) != 0:
            self.position_capital = float(self.binance.futures_account()['totalMarginBalance'])*leverage
            trades_to_close, final_trades_to_enter = self._profit_optimization(essential_metrics, current_positions, trades_to_enter, number_of_trades)
            
            if trades_to_close:
                self.order_history = self.order_history.append(executor.exit_market(self.binance, self.get_positions_amount(trades_to_close), reason='lower rank'))
                display(self.close_trade_info)
                current_positions = self.get_current_trade_progress(printout=False)
            if trades_to_enter:
                dividing_factor = (len(current_positions) + len(trades_to_enter))/len(current_positions)
                positions_amount = self.get_positions_amount(current_positions['symbol'].values, divide_by=dividing_factor, max_trades=number_of_trades)
                if positions_amount:
                    self.order_history = self.order_history.append(executor.exit_market(self.binance, positions_amount, reason='making space'))
                self.position_capital = float(self.binance.futures_account()['totalMarginBalance'])*leverage
                self.order_history = self.order_history.append(executor.enter_market(self.binance, final_trades_to_enter, self.position_capital, number_of_trades, reason='higher rank'))
                display(self.order_history)
            else:
                print("No Trades made")
        self.order_history.to_csv("FAB Order History.csv")
        self.trade_replay.append(self.get_current_trade_progress(printout=False))

    def close_any_old_trades(self, screener, executor, current_positions, tfs):
        if len(current_positions) == 0:
            return False
        trades_to_close = self._check_trade_close(screener, current_positions, tfs)
        if trades_to_close:
            self.order_history = self.order_history.append(executor.exit_market(self.binance, self.get_positions_amount(trades_to_close), reason='trade closed'))

    def update_trades_to_enter(self, current_positions, trades_to_enter):
        if len(trades_to_enter) ==0:
            trades_to_enter = [(symbol, self.executor.dic[side], None) for symbol, side in current_positions[['symbol','side']].values]
        return trades_to_enter

    def monitor_fab(self, screener: Screener, df_metrics, tfs, number_of_trades=3, leverage=0.001, recency=-1):
        # self.order_history = pd.DataFrame()
        self.load_account()
        executor = self.executor
        self.capital = self.get_capital()
        self.leverage = leverage
        self.load_account()
        essential_metrics = Helper.drop_extra_delays(df_metrics, screener.tf_delay_match)

        now = Helper.current_minute_datetime()
        while True:
            if datetime.now() >= now + pd.Timedelta(1, 'minute'):
                clear_output(wait=True)
                print("Current time: \t", datetime.now(), end='\n\n')

                current_positions = self.get_current_trade_progress()

                self.close_any_old_trades(screener, executor, current_positions, tfs)

                self.remaining_capital = self.calculate_remaining_capital(current_positions, self.capital, leverage)

                ## trades_to_enter is a list of lists (Ex. [('BANDUSDT', 'Short'), ('BCHUSDT', 'Short')]
                ## df_recent_signals is a regular screener dataframe
                trades_left = number_of_trades - len(current_positions)
                trades_to_enter, df_recent_signals = screener.top_trades(trader=self, df_metrics=df_metrics, tfs=tfs, n=trades_left, recency=recency)
                self.trades_to_enter, self.df_recent_signals = trades_to_enter, df_recent_signals

                print("Signals:")
                display(df_recent_signals)

                if self.remaining_capital > 50 and trades_left > 0 and trades_to_enter:
                    self.trade_free_capital(executor, self.leverage, self.remaining_capital, trades_to_enter)

                elif self.remaining_capital >50 and len(current_positions) != 0:
                    trades_to_enter = self.update_trades_to_enter(current_positions=current_positions, trades_to_enter=trades_to_enter)
                    self.trade_free_capital(executor, self.leverage, self.remaining_capital, trades_to_enter)
                else:
                    print("Leverage: ", self.leverage)
                    print("Trades to Enter: ", trades_to_enter)
                    print("Max Trades: ", number_of_trades)
                    self.optimize_trades(executor, current_positions, self.leverage, essential_metrics, trades_to_enter, number_of_trades)
                Helper.sleep(60)
                now = Helper.current_minute_datetime()
                print()
            self.output_loading()



