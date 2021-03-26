from binance.client import Client
import pandas as pd
from datetime import datetime
from helper import Helper
from fab_strategy import FabStrategy
from dataloader import _DataLoader
from trading_history import TradeHistory
import math
import time
import numpy as np
import os


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

    def __init__(self):
        self.start = False
        self.client = None
        self.capital = None
        self.leverage = 1 / 1000
        self.tf = None
        self.symbol = None
        self.loader = _DataLoader()
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
        local_path = os.path.dirname(os.path.dirname(os.path.abspath("__file__"))) + r"\local\binance_api.txt"
        info = pd.read_csv(local_path).set_index('Name')
        API_KEY = info.loc["API_KEY", "Key"]
        SECRET = info.loc["SECRET", "Key"]
        self.client = Client(API_KEY, SECRET)
        self.capital = int(float(self.client.futures_account_balance()[0]['balance'])) + additional_balance
        return "Welcome Haseab"

    def set_leverage(self, leverage: float) -> float:
        """Sets the current leverage of the account: should normally be 1. And 0.001 for testing purposes"""
        self.leverage = leverage
        return self.leverage

    def set_timeframe(self, tf: int) -> int:
        """Sets the timeframe of the trading data"""
        self.tf = tf
        return self.tf

    def get_necessary_data(self, symbol: str, tf: int, max_candles_needed: int) -> pd.DataFrame:
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
        ranges = Helper.determine_candle_positions(max_candles_needed, tf)

        for i in range(len(ranges)):
            try:
                df = df.append(self.loader._get_binance_futures_candles(symbol, int(ranges[i]), int(ranges[i + 1]), now))
            except IndexError:
                pass
        return df.drop_duplicates()

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
        last_few_candles['Datetime'] = Helper.millisecond_timestamp_to_datetime(last_few_candles.index)
        last_few_candles = last_few_candles[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

        # Abstracting minute data into appropriate tf
        last_few_candles = self.loader._timeframe_setter(last_few_candles, self.tf, 0)

        # Updating original data with new data.
        self.df = self.df.append(last_few_candles).drop_duplicates()

    def get_position_amount(self, symbol: str) -> float:
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
        for coin_futures_info in self.client.futures_position_information():
            if coin_futures_info['symbol'] == symbol:
                return float(coin_futures_info["positionAmt"])
        return None

    def set_asset(self, symbol: str, tf: int, max_candles_needed: int = 231) -> str:
        """
        Set Symbol of the ticker and load the necessary data with the given timeframe to trade it

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
                self.client.futures_klines(symbol=symbol, interval=map_tf[tf], startTime=start_time))
            df.drop(df.tail(1).index, inplace=True)
        else:
            # If it doesn't match Binance available timeframes, it must be transformed after fetching 1m data.
            df = self.get_necessary_data(symbol, tf, max_candles_needed)
            df = self.loader._timeframe_setter(df, tf)

        # Adding Datetime column for readability of the timestamp. Also more formatting done
        df['Datetime'] = Helper.millisecond_timestamp_to_datetime(df.index)
        df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
        df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

        self.df = df
        self.symbol = symbol

        return f"Symbol changed to {self.symbol}"

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
        client = self.client
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
