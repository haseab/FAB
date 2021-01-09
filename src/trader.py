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


class Trader():
    """
    Purpose is to trade live on the Binance Crypto Exchange

    Attributes
    -----------
    symbol: str
    client: Client object - From binance.py
    capital: float - represents how much money is in account
    leverage: float - how many multiples of your money you want to trade
    live_trade_history: list of lists - trading data
    trade_journal: pd.DataFrame - dataframe with all raw data from binance response
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
        self.symbol = None
        self.client = None
        self.capital = None
        self.leverage = 1 / 1000
        self.live_trade_history = TradeHistory()
        self.trade_journal = pd.DataFrame()
        self.tf = None
        self.df = None
        self.loader = _DataLoader()

    def load_account(self) -> str:
        """
        Sign in to account using API_KEY and using Binance API
        """
        info = pd.read_csv(r'C:\Users\haseab\Desktop\Python\PycharmProjects\FAB\local\binance_api.txt').set_index(
            'Name')
        API_KEY = info.loc["API_KEY", "Key"]
        SECRET = info.loc["SECRET", "Key"]
        self.client = Client(API_KEY, SECRET)
        # Initializing how much money I have on the exchange. Sample 20000 has been added.
        self.capital = int(float(self.client.futures_account_balance()[0]['balance'])) + 20000
        return "Welcome Haseab"

    def set_leverage(self, leverage: float) -> float:
        """Sets the current leverage of the account: should normally be 1. And 0.001 for testing purposes"""
        self.leverage = leverage
        return self.leverage

    def set_timeframe(self, tf: int) -> int:
        """Sets the timeframe of the trading data"""
        self.tf = tf
        return self.tf

    def get_necessary_data(self, symbol: str, tf: int) -> pd.DataFrame:
        """
        Gets the minimum necessary data to trade this asset. Only a symbol and timeframe need to be inputted

        Note: This method is used as a way to tackle the 1000 candle limit that is currently on the Binance API.
        A discrete set of ~1000 group candles will be determined, and then the data will be extracted from each,
        using the _get_binance_futures_candles method, and then all of them will be merged together.

        Parameters:
        symbol: str     Ex. "BTCUSDT", "ETHUSDT"
        tf: int         Ex. 1, 3, 5, 77, 100

        :return pd.DataFrame of candlestick data

        """
        now = time.time()

        # The 231 MA needs 231 candles of data to work. We use 4 more candles for safety.
        maximum_candles_needed = 235

        # Formula for determining how many discrete sets are needed
        split_number = math.ceil(tf * maximum_candles_needed / 1000) + 1

        # Determining the exact indices of when the set boundaries end
        ranges = np.ceil(np.linspace(0, tf * 235, num=split_number))
        ranges = [int(i) for i in ranges[::-1]]

        df = pd.DataFrame()

        # Grabbing each set of about 1000 candles and appending them one after the other
        for i in range(len(ranges)):
            try:
                df = df.append(
                    self.loader._get_binance_futures_candles(symbol, int(ranges[i]), int(ranges[i + 1]), now))
            except IndexError:
                pass
        return df.drop_duplicates()

    def _update_data(self, diff: int) -> None:
        """
        Used to update the trading data with the exchange data so that it is real time

        Parameters:
        -----------
        diff: a number that explains how many minutes of disparity there is between current data and live data.

        :return dataframe with the most up-to-date data.
        """
        # Calculating how many minutes to fetch from API
        minutes = math.floor(diff) + 1

        # Getting minute candlestick data. Number of minute candlesticks represented by "minutes" variable
        last_price = self.loader._get_binance_futures_candles("BTCUSDT", minutes)
        last_price['Datetime'] = [datetime.fromtimestamp(i / 1000) for i in last_price.index]
        last_price = last_price[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

        # Abstracting minute data into appropriate tf
        last_price = self.loader._timeframe_setter(last_price, self.tf, 0)

        # Updating original data with new data.
        self.df = self.df.append(last_price).drop_duplicates()

    def get_position(self, symbol: str) -> float:
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
        return float([i["positionAmt"] for i in self.client.futures_position_information() if i['symbol'] == symbol][0])

    def set_asset(self, symbol: str) -> str:
        """
        Set Symbol of the ticker and load the necessary data with the given timeframe to trade it

        Parameters:
        ------------
        symbol: str     Ex. "BTCUSDT", "ETHUSDT"

        :return str response
        """
        self.symbol = symbol
        # For Binance API purposes, 240 min needs to be inputted as "4h" on binance when fetching data
        map_tf = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h", 360: "6h", 480: "8h"}

        startTime = int((time.time() - self.tf * 235 * 60) * 1000)

        if self.tf in [1, 3, 5, 15, 30, 60, 120, 240, 360, 480]:  # Note: 12H, 1D, 3D, 1W, 1M are also recognized

            # Fetching data from Binance if it matches the eligible timeframe, as it will be faster
            self.df = Helper.into_dataframe(
                self.client.futures_klines(symbol=symbol, interval=map_tf[self.tf], startTime=startTime))
            self.df.drop(self.df.tail(1).index, inplace=True)

        else:
            # If it doesn't match Binance available timeframes, it must be transformed after fetching 1m data.
            self.df = self.get_necessary_data(symbol, self.tf)
            self.df = self.loader._timeframe_setter(self.df, self.tf)

        # Adding Datetime column for readability of the timestamp. Also more formatting done
        self.df['Datetime'] = [datetime.fromtimestamp(i / 1000) for i in self.df.index]
        self.df = self.df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
        self.df[["Open", "High", "Low", "Close", "Volume"]] = self.df[
            ["Open", "High", "Low", "Close", "Volume"]].astype(float)

        return f"Symbol changed to {self.symbol}"

    def make_row(self, high=[], low=[], volume=[], count: int = 0, open_price: float = None, open_date: str = None) -> (
    pd.DataFrame, list, list, list, list, list):
        """
        Helper function used to update the last row of a dataframe, using all of the data in the previous "last row" as input.
        This is used to update the price of the candlestick live, with the incomplete current candle constantly updating.

        Parameters:
        -----------
        high:       list        Ex. [23348, 23350, 23335, 23330, 23339]
        low:        list        Ex. [23300, 23345, 23335, 23320, 23300]
        volume:     list        Ex. [47, 31, 110, 117, 2, 55]
        count:      int         Ex. 1,2,3
        open_price: float       Ex. 23342
        open_date:  str         Ex. "2020-08-04 17:33:02"

        Returns tuple -> (pd.DataFrame, list, list, list, list, list)
        """
        # row variable gets a pd.DataFrame of size 1.
        row = self.loader._get_binance_futures_candles("BTCUSDT", 1).reset_index()
        timestamp = row.loc[0, "Timestamp"]
        close = float(row.loc[0, "Close"])
        high.append(float(row.loc[0, "High"]))
        low.append(float(row.loc[0, "Low"]))
        volume.append(float(row.loc[0, "Volume"]))

        # Initial values of a candle that only get set on the first iteration.
        if count == 0:
            open_price = row.loc[0, "Open"]
            open_date = timestamp

        # Adding to previous candlestick data of the last row by updating the row.
        dfrow = pd.DataFrame([[open_date, datetime.fromtimestamp(open_date / 1000), open_price, max(high), min(low),
                               close, sum(volume)]], \
                             columns=["Timestamp", "Datetime", "Open", "High", "Low", "Close", "Volume"])

        dfrow[["Open", "High", "Low", "Close", "Volume"]] = dfrow[["Open", "High", "Low", "Close", "Volume"]].astype(
            float)
        dfrow = dfrow.set_index("Timestamp")

        return dfrow, high, low, volume, open_price, open_date

    def load__existing_asset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sets the trading data to an already existing dataframe, passed in as an argument"""
        self.df = df
        return self.df

    def start_trading(self, strategy: FabStrategy, sensitivity=0, debug=False) -> None:
        """
        Starts the process of trading live. Each minute, the last row of the data is updated and Rule #2 is checked,
        otherwise, it waits till the end of the timeframe where it gets the latest data before it checks the rest of the rules.
        If it does see something following the rules, it will buy/short, given the initial parameters it has. (e.g. leverage, quantity)
        This process continues indefinetly, unless interrupted.

        Returns None.
        """
        executor = TradeExecutor()
        count, open_price = 0, 0
        open_date = None
        high, low, volume = [], [], []

        # Turning trading bot "on"
        self.start = True

        # Loading data into strategy, and creating moving averages.
        strategy.load_data(self.df)
        strategy.create_objects()

        while self.start != False:

            diff = Helper.calculate_minute_disparity(self.df, self.tf)
            if round(time.time() % 60, 1) == 0 and diff <= self.tf:

                # Getting the most up-to-date row of the <tf>-min candelstick
                dfrow, high, low, volume, open_price, open_date = self.make_row(high, low, volume, count, open_price,
                                                                                open_date)

                if debug == True:
                    print(dfrow)
                    print(f"{self.tf - diff} minutes left")

                # Updating moving averages
                strategy.load_data(self.df.append(dfrow))
                strategy.create_objects()

                # Checking only Rule 2, because it needs a minute by minute check.
                # Second condition is making sure that there is no existing position
                if strategy.rule_2_buy_enter(-1, sensitivity) and self.get_position(self.symbol) == 0:
                    trade_info = executor.enter_market(self.symbol, "BUY", 2)
                elif strategy.rule_2_short_enter(-1, sensitivity) and self.get_position(self.symbol) == 0:
                    trade_info = executor.enter_market(self.symbol, "SELL", 2)

                # Saves CPU usage, waits 5 seconds before the next minute
                time.sleep(55)

                count += 1

            elif diff > self.tf:
                # Choosing to update data using Binance API instead of appending the completed final row from the while loop
                self._update_data(math.floor(diff))

                # Updating Moving averages
                strategy.load_data(self.df)
                strategy.create_objects()

                # Checks for the rest of the rules
                if strategy.rule_2_short_stop(-1) and self.get_position(
                        self.symbol) < 0 and self.TradeHistory.last_trade().rule == "Rule 2":
                    trade_info = executor.exit_market(self.symbol, 2, self.get_position(self.symbol))
                elif strategy.rule_2_buy_stop(-1) and self.get_position(
                        self.symbol) > 0 and self.live_trade_history.last_trade().rule == "Rule 2":
                    trade_info = executor.exit_market(self.symbol, 2, self.get_position(self.symbol))
                elif strategy.rule_1_buy_enter(-1) and self.get_position(self.symbol) == 0:
                    trade_info = executor.enter_market(self.symbol, "BUY", 1)
                elif strategy.rule_1_buy_exit(-1) and self.get_position(self.symbol) > 0:
                    trade_info = executor.exit_market(self.symbol, 1, self.get_position(self.symbol))
                elif strategy.rule_1_short_enter(-1) and self.get_position(self.symbol) == 0:
                    trade_info = executor.enter_market(self.symbol, "SELL", 1)
                elif strategy.rule_1_short_exit(-1) and self.get_position(self.symbol) < 0:
                    trade_info = executor.exit_market(self.symbol, 1, self.get_position(self.symbol))
                elif strategy.rule_3_buy_enter(-1) and self.get_position(self.symbol) == 0:
                    trade_info = executor.enter_market(self.symbol, "BUY", 3)
                elif strategy.rule_3_short_enter(-1) and self.get_position(self.symbol) == 0:
                    trade_info = executor.enter_market(self.symbol, "SELL", 3)

                    # Resetting candlesticks
                high, low, volume = [], [], []
                count = 0
                time.sleep(1)

                if debug == True:
                    print("next")
