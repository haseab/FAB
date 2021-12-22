import math
import os
import random
import time
from datetime import datetime

import dateparser
import numpy as np
import pandas as pd
from binance.client import Client
from decouple import config
from ftx import FtxClient
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

    def __init__(self, qtrade=False, db=False, ib=False):
        self.start = False
        self.binance = Client()
        self.ftx = FtxClient()
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

        API_KEY_FTX = config('API_KEY_FTX')
        API_SECRET_FTX  = config('API_SECRET_FTX')

        self.binance = Client(api_key=API_KEY_BINANCE, api_secret=API_SECRET_BINANCE)
        self.ftx = FtxClient(api_key=API_KEY_FTX, api_secret=API_SECRET_FTX)
        return "Connected to Both Binance and FTX"

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

    def get_necessary_data(self, symbol: str, tf: int, max_candles_needed:int = 235) -> pd.DataFrame:
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

    def get_positions_amount(self, list_of_symbols: list) -> list:
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
            for coin_futures_info in self.binance.futures_position_information():
                if coin_futures_info['symbol'] == symbol:
                    position_values[symbol] = float(coin_futures_info["positionAmt"])
        return position_values

    def set_asset_v2(self, symbol: str, tf: int, max_candles_needed: int = 231) -> pd.DataFrame:
        binance_tf = Helper.find_greatest_divisible_timeframe(tf)
        df = self.get_necessary_data(symbol, binance_tf, max_candles_needed*(tf//binance_tf))
        self.df, self.tf, self.symbol= df, tf, symbol
        return df

    def set_asset(self, symbol: str, tf: int, max_candles_needed: int = 231, keep_last_row=False) -> pd.DataFrame:
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
            if drop_last_row:
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

        tf_candle = self.loader._timeframe_setter(minute_candles, tf, drop_last_row=False)
        return tf_candle

    def load__existing_asset(self, df: pd.DataFrame) -> pd.DataFrame:
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
                pnl = Helper.calculate_short_profitability(enter_price, last_price, 0.99975)
            elif side == 'BUY':
                pnl = Helper.calculate_long_profitability(enter_price, last_price, 0.99975)
            usd_size = abs(round(position_size * enter_price, 2))
            pnl_usd = usd_size*(pnl-1)
            trade_progress = trade_progress.append(pd.DataFrame([[symbol, side, enter_price, abs(position_size), usd_size, last_price, pnl, pnl_usd]],
                                                            columns = ['symbol', 'side', 'enter_price', 'size', 'usd size', 'current_price', 'pnl (%)' , 'pnl (USD)']),
                                                ignore_index=True)
        if printout:
            display(trade_progress)

        self.trade_progress = trade_progress
        return trade_progress

    def profit_optimization(self, signals, trade_progress_partial, screener):
        print()
        print("Optimizing Trades:... ")
        trades_to_close, trades_to_enter = [], []
        trade_metrics = self.trade_metrics if type(self.trade_metrics) != type(None) else pd.read_csv("Current metrics for current_trades.csv").set_index('symbol')

        if len(trade_progress_partial) == 0 or len(trade_metrics) == 0:
            return [], []

        signals = self.drop_existing_enters(signals, trade_progress_partial)
        potentials = signals['peak_unrealized_profit'].sort_values(ascending=False)


        trade_metrics[['date', 'due date']] = trade_metrics[['date', 'due date']].astype('datetime64')
        trade_progress = trade_progress_partial.merge(trade_metrics.reset_index(), on='symbol').set_index('symbol')
        trade_progress['delta profit'] = trade_progress['peak_unrealized_profit'] - trade_progress['pnl']
        trade_progress['candles left'] = ((trade_progress['due date'] - datetime.now()).dt.total_seconds() // trade_progress['tf']).astype('int64')

        if len(signals) == 0:
            trades_to_close = self.check_trade_close(screener, trade_progress, trades_to_close)

        trades_to_close, trades_to_enter = self.check_other_signals(potentials, signals, trade_progress, trades_to_close, trades_to_enter)

        self.new2 = signals, potentials, trade_metrics, trades_to_close, trades_to_enter

        return self.check_enter_exit_trap(trades_to_close, trades_to_enter)

    def drop_existing_enters(self, signals, trade_progress_partial):
        for symbol, tf in signals.index:
            if symbol in trade_progress_partial['symbol'].values:
                signals = signals.drop((symbol, tf))
        return signals

    def check_enter_exit_trap(self, trades_to_close, trades_to_enter):
        for symbol, tf in trades_to_enter:
            if symbol in trades_to_close:
                trades_to_enter.remove((symbol, tf))
                trades_to_close.remove(symbol)
        return trades_to_close, trades_to_enter

    def update_current_trade_metrics(self):
        df_all_signals = pd.read_csv("Recent signals.csv").set_index(["symbol", "tf"])
        df_all_signals['date'] = df_all_signals['date'].astype('datetime64')
        trade_metrics = pd.DataFrame()
        trade_progress = self.monitor_trades(printout=False)
        self.trade_progress = trade_progress

        if len(df_all_signals) == 0 or len(trade_progress)==0:
            return None

        for symbol, tf in df_all_signals.index:
            if symbol in trade_progress['symbol'].values:
                trade_metrics = trade_metrics.append(df_all_signals.loc[[(symbol, tf)]])

        trade_metrics.to_csv("Current metrics for current_trades.csv")
        display(trade_metrics)

        self.trade_metrics = trade_metrics
        return trade_metrics

    # def update_current_trade_metrics(self, potentials, trade_metrics, trades_to_close, trades_to_enter):
    #     for symbol, tf in trade_metrics.index:
    #         if symbol in trades_to_close:
    #             trade_metrics.drop((symbol, tf), inplace=True)
    #     for symbol, tf in potentials.index:
    #         if (symbol, tf) in trades_to_enter:
    #             trade_metrics = trade_metrics.append(potentials.loc[(symbol, tf), :])
    #     print(trades_to_close, trades_to_enter)
    #     display(trade_metrics)
    #     trade_metrics.to_csv("Current metrics for current_trades.csv")
    #     self.trade_metrics = trade_metrics

    def check_other_signals(self, potentials, signals, trade_progress, trades_to_close, trades_to_enter):
        for symbol, tf in potentials.index:
            if len(trade_progress['delta profit']) == 0:
                break

            if potentials.loc[(symbol, tf)] - 1 > min(trade_progress['delta profit']) + 0.0075:
                side = signals.loc[(symbol, tf), 'signal'][1]
                index_of_trade = trade_progress['delta profit'].idxmin()

                trades_to_close.append(index_of_trade)
                print('trades switched')
                trade_progress.drop(index_of_trade, inplace=True)
                trades_to_enter.append((symbol, side))
        return trades_to_close, trades_to_enter

    def score_calculator(self, df_current, df_length, date=False, gain=False):
        weight_dic = {}
        if date:
            weight_dic_x = np.round(np.linspace(0, 100, 1001), 1)
            weight_dic_y = np.round(1.03 ** (weight_dic_x - 80), 5)
            weight_dic = dict(zip(weight_dic_x, weight_dic_y))

        elif gain:
            weight_dic_neg_x = np.round(np.linspace(-50, 0.1, 500), 1)
            weight_dic_neg_y = np.round(15/(weight_dic_neg_x**2 + 15), 5)
            weight_dic_pos_x = np.round(np.linspace(0, 100, 1001), 1)
            weight_dic_pos_y = np.round(0.985**weight_dic_pos_x, 5)
            weight_dic_x = np.append(weight_dic_neg_x, weight_dic_pos_x)
            weight_dic_y = np.append(weight_dic_neg_y, weight_dic_pos_y)
            weight_dic = dict(zip(weight_dic_x, weight_dic_y))

        if not weight_dic:
            raise Exception("date and gain are both set to False. Set one to true")

        relative_current = np.round((1-df_current/df_length)*100, 1)
        score = pd.Series([weight_dic[value] for value in relative_current])
        return score * df_current

    def check_trade_close(self, screener, trade_progress, trades_to_close, max_candle_history=10):
        trade_progress = trade_progress.reset_index().drop_duplicates(subset='symbol').set_index('symbol')
        for symbol in trade_progress.index:
            tf = trade_progress.loc[symbol, 'tf']
            df = self.set_asset(symbol=symbol, tf=tf, max_candles_needed=231 + max_candle_history)
            close_signal, *other = screener.check_for_signals(df, self.strategy,
                                                              max_candle_history=max_candle_history, exit=True)
            # print(close_signal)
            if close_signal:
                trades_to_close.append(symbol)
                print('trade closed from rules')
                continue
            tf = trade_progress.loc[symbol, 'tf']
            elapsed_candles = int(
                (datetime.now() - trade_progress['date'].loc[symbol]).total_seconds() // (60 * tf) + 1)

            if elapsed_candles > trade_progress['num_candles_to_peak'].loc[symbol]:
                trades_to_close.append(symbol)
                print('trade closed from exceeding candles')
        return trades_to_close

    def _check_trade_close(self, screener: Screener, current_positions, tfs):
        trades_to_close = []
        current_position_symbols = current_positions['symbol'].values
        # current_position_dfs = {symbol:screener.clean_results[symbol] for symbol in current_position_symbols}
        for symbol in current_position_symbols:
            for tf in tfs:
                df = screener.df_dic[(symbol, tf)]
                close_signal, *other = screener._check_for_tf_signals(df, max_candle_history=10, exit=True)
                if close_signal:
                    trades_to_close.append(symbol)
                    print(f'{symbol} Trade Closed')
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
        print(f"Free capital: {remaining_to_invest} USD")
        self.df_orders = executor.enter_market(self.binance, symbol_side_pair=trades_to_enter,
                                               capital=remaining_to_invest,
                                               leverage=leverage)
        display(self.df_orders)

    def calculate_remaining_capital(self, current_positions, capital):
        total_invested = current_positions['usd size'].sum() if len(current_positions) > 0 else 0
        remaining_to_invest = capital - total_invested
        return remaining_to_invest

    def optimize_trades(self, executor, current_positions, leverage, df_metrics, trades_to_enter, number_of_trades):
        if len(current_positions) != 0:
            self.capital = current_positions['usd size'].sum()
            trades_to_close, final_trades_to_enter = self._profit_optimization(df_metrics, current_positions, trades_to_enter, number_of_trades)
            if trades_to_close:
                self.close_trade_info = executor.exit_market(self.binance, self.get_positions_amount(trades_to_close))
                display(self.close_trade_info)
            if trades_to_enter:
                self.enter_trade_info = executor.enter_market(self.binance, final_trades_to_enter, self.capital, leverage=leverage)
                display(self.enter_trade_info)
            else:
                print("No Trades made")

    def close_any_old_trades(self, screener, executor, current_positions, tfs):
        trades_to_close = self._check_trade_close(screener, current_positions, tfs)
        if trades_to_close:
            self.close_trade_info = executor.exit_market(self.binance, self.get_positions_amount(trades_to_close))

    def _profit_optimization(self, df_metrics, current_positions, trades_to_enter, number_of_trades):
        """
        Returns:
            final_trades_to_enter: Ex. [('ZILUSDT', "Short"), ('BTCUSDT', 'Long'), ("ETHUSDT", "Short")]
            trades_to_close:       Ex. ['ADAUSDT']
        """
        print()
        print("Optimizing Trades:... ")
        trades_of_interest = [symbol for symbol, side in trades_to_enter]
        ranks = df_metrics[df_metrics['amount of data'] >= 5].sort_values('avg pnl', ascending=False)
        key = ranks.groupby('symbol').max().sort_values('avg pnl', ascending=False)
        current_positions_symbols = list(current_positions['symbol'].values)

        all_symbols = list(set(trades_of_interest).union(set(current_positions_symbols)))
        top_x_symbols = sorted(all_symbols, key=lambda x: key.loc[x]['avg pnl'], reverse=True)[:number_of_trades]
        
        partial_trades_to_enter = list(set(top_x_symbols).intersection(set(trades_of_interest)).difference(set(current_positions_symbols)))
        trades_to_close = list(set(current_positions_symbols).difference(set(top_x_symbols)))

        final_trades_to_enter = [(symbol, side) for symbol, side in trades_to_enter if symbol in partial_trades_to_enter]

        return trades_to_close, final_trades_to_enter

    def monitor_fab(self, screener: Screener, df_metrics, tfs, number_of_trades=3, leverage=0.001, recency=-1):
        self.load_account()
        executor = self.executor
        self.capital = self.get_capital()
        self.leverage = leverage

        now = Helper.current_minute_datetime()
        while True:
            if datetime.now() >= now + pd.Timedelta(1, 'minute'):
                clear_output(wait=True)
                print(datetime.now())

                current_positions = self.get_current_trade_progress()
                self.close_any_old_trades(screener, executor, current_positions, tfs)

                remaining_to_invest = self.calculate_remaining_capital(current_positions, self.capital)

                ## trades_to_enter is a list of lists (Ex. [('BANDUSDT', 'Short'), ('BCHUSDT', 'Short')]
                ## df_recent_signals is a regular screener dataframe
                trades_left = number_of_trades - len(current_positions)
                trades_to_enter, df_recent_signals = screener.top_trades(trader=self, df_metrics=df_metrics, tfs=tfs, n=trades_left)

                if remaining_to_invest*self.leverage > 50 and trades_to_enter:
                    self.trade_free_capital(executor, self.leverage, remaining_to_invest*self.leverage, trades_to_enter)
                else:
                    self.optimize_trades(executor, current_positions, self.leverage, df_metrics, trades_to_enter, number_of_trades)

                now = Helper.current_minute_datetime()
                print()
            self.output_loading()



