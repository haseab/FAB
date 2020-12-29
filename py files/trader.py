from binance.client import Client
import pandas as pd
from dataloader import DataLoader
from datetime import datetime, timedelta
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
    get_postion
    enter_market
    exit_market
    stop_market
    enter_limit
    exit_limit
    get_last_x_candles
    get_necessary_data
    set_asset
    make_row
    load_asset
    start_trading

    Please look at each method for descriptions
    """

    def __init__(self):
        self.symbol = None
        self.client = None
        self.capital = None
        self.leverage = 1 / 1000
        self.live_trade_history = ["List of Trades"]
        self.trade_journal = pd.DataFrame()
        self.tf = None
        self.df = None

    def _update_data(self, diff: int) -> pd.DataFrame:
        """
        Used to update the trading data with the exchange data so that it is real time

        Parameters:
        -----------
        diff: a number that explains how many minutes of disparity there is between current data and live data.

        Returns: dataframe with the most up-to-date data.
        """
        minutes = math.floor(diff) + 1
        last_price = pd.DataFrame(self.client.futures_klines(symbol=self.symbol, interval="1m",
                                                             startTime=(int(time.time()) - 60 * (minutes)) * 1000),
                                  columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "Timestamp_end", "",
                                           "", "", "", ""]).set_index("Timestamp")
        last_price['Datetime'] = [datetime.fromtimestamp(i / 1000) for i in last_price.index]
        last_price = last_price.append(last_price.tail(1))
        last_price = last_price[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
        load1 = DataLoader()
        last_price = load1.timeframe_setter(last_price, self.tf, 0)
        self.df = self.df.append(last_price).drop_duplicates()

    def load_account(self) -> str:
        """
        Sign in to account using API_KEY and using Binance API
        """
        info = pd.read_csv(r'C:\Users\owner\Desktop\Python\PycharmProjects\FAB\local\binance_api.txt').set_index('Name')
        API_KEY = info.loc["API_KEY", "Key"]
        SECRET = info.loc["SECRET", "Key"]
        self.client = Client(API_KEY, SECRET)
        self.capital = int(float(self.client.futures_account_balance()[0]['balance'])) + 20000
        return "Welcome Haseab"

    def set_leverage(self, leverage: float) -> float:
        """Sets the current leverage of the account: should normally be 1 and 0.001 for testing purposes"""
        self.leverage = leverage
        return self.leverage

    def set_timeframe(self, tf: int) -> int:
        """Sets the timeframe of the trading data"""
        self.tf = tf
        return self.tf

    def get_position(self, symbol: str) -> float:
        """
        Gets the total amount of current position for a given symbol
        Parameters:
        ------------
        symbol: str
            Ex. "BTCUSDT"
                "ETHUSDT"

        returns float
            Ex. If a 1.000 BTC position is open, it will return 1.0
        """
        return float([i["positionAmt"] for i in self.client.futures_position_information() if i['symbol'] == symbol][0])

    def enter_market(self, symbol: str, side: str, leverage: float, rule_no: int) -> list:
        """
        Creates a order in the exchange, given the symbol

        Parameters:
        ------------
        symbol: str       Ex. "BTCUSDT", "ETHUSDT"
        side: str         Ex. 'BUY', SELL'
        leverage: float   Ex. 0.001, 0.500, 1.000
        rule_no: int      Ex.  1, 2, 3

        returns: dict
            Ex. {'orderId': 11012154287,
                 'symbol': 'BTCUSDT',
                 'status': 'FILLED',
                 'clientOrderId': '2kiNaS1NdYG6QfCGjyIE1B',
                 'price': '0',
                 'avgPrice': '27087.68000',
                 'origQty': '0.001',
                 'executedQty': '0.001',
                 'cumQuote': '27.08768',
                 'timeInForce': 'GTC',
                 'type': 'MARKET',
                 'reduceOnly': False,
                 'closePosition': False,
                 'side': 'SELL',
                 'positionSide': 'BOTH',
                 'stopPrice': '0',
                 'workingType': 'CONTRACT_PRICE',
                 'priceProtect': False,
                 'origType': 'MARKET',
                 'time': 1609197914670,
                 'updateTime': 1609197914825}
        """
        minutes = 2
        last_price = float(self.client.futures_klines(symbol=symbol, interval="1m",
                                                      startTime=(int(time.time()) - 60 * (minutes)) * 1000)[-1][4])
        #         last_price = float(self.client.get_historical_klines(symbol=symbol, interval="1m", start_str="150 seconds ago UTC")[-1][4])
        #         assert (round(sig_fig(self.capital*leverage/(last_price),4),3)) < 0.1
        enterMarketParams = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': round(sig_fig(self.capital * leverage / (last_price), 4), 3)
        }
        self.latest_trade = self.client.futures_create_order(**enterMarketParams)
        self.latest_trade_info = self.client.futures_get_order(symbol=symbol, orderId=self.latest_trade["orderId"])
        date = str(datetime.fromtimestamp(self.latest_trade_info['time'] / 1000))[:19]
        dic = {"BUY": "Long", "SELL": "Short"}
        self.live_trade_history.append(
            [dic[side], "Enter", date, self.latest_trade_info['avgPrice'][:8], f"Rule {rule_no}"])
        self.trade_journal = self.trade_journal.append([self.latest_trade_info])
        return self.latest_trade_info

    def exit_market(self, symbol: str, rule_no: int, position_amount: float) -> list:
        """
        Considers the current position you have for a given symbol, and closes it accordingly

        Parameters:
        ------------
        symbol: str       Ex. "BTCUSDT", "ETHUSDT"
        rule_no: int      Ex.  1, 2, 3

        returns: dict     Ex. see "enter_market" description
        """
        side = "BUY" if position_amount < 0 else "SELL"
        exitMarketParams = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': abs(position_amount)
        }
        self.latest_trade = self.client.futures_create_order(**exitMarketParams)
        self.latest_trade_info = self.client.futures_get_order(symbol=symbol, orderId=self.latest_trade["orderId"])
        date = str(datetime.fromtimestamp(self.latest_trade_info['time'] / 1000))[:19]
        dic = {"BUY": "Short", "SELL": "Long"}
        self.live_trade_history.append(
            [dic[side], "Exit", date, self.latest_trade_info['avgPrice'][:8], f"Rule {rule_no}"])
        self.trade_journal = self.trade_journal.append([self.latest_trade_info])
        return self.latest_trade_info

    def stop_market(self, symbol: str, price: float, position_amount: float) -> list:
        """
        Sets a stop loss (at market) at a given price for a given symbol

        Parameters:
        ------------
        symbol: str            Ex. "BTCUSDT", "ETHUSDT"
        price: float           Ex. 0.001, 0.500, 1.000
        position_amount: int   Ex.  1.0, 2000, 153.5

        returns: dict          Ex. see "enter_market" desc

        """
        side = "BUY" if position_amount < 0 else "SELL"
        stopMarketParams = {
            'symbol': symbol,
            'side': side,
            'type': 'STOP_MARKET',
            'stopPrice': price,
            'quantity': abs(position_amount)
        }
        self.latest_trade = self.client.futures_create_order(**stopMarketParams)
        self.latest_trade_info = self.client.futures_get_order(symbol=symbol, orderId=self.latest_trade["orderId"])
        return self.latest_trade_info

    def enter_limit(self, symbol: str, side: str, price: float, leverage: float) -> list:
        """
        Sets a limit order at a given price for a given symbol

        Parameters:
        ------------
        symbol: str       Ex. "BTCUSDT", "ETHUSDT"
        side: str         Ex. 'BUY', SELL'
        leverage: float   Ex. 0.001, 0.500, 1.000
        price: int        Ex.  10254, 530, 1.01

        returns: dict     Ex. see "enter_market" desc

        """
        enterLimitParams = {
            'symbol': symbol,
            'side': side,
            'type': "LIMIT",
            'price': price,
            'timeInForce': "GTC",
            'quantity': round(sig_fig(self.capital * leverage / (last_price), 4), 3)
        }
        self.latest_trade = self.client.futures_create_order(**enterLimitParams)
        self.latest_trade_info = self.client.futures_get_order(symbol=symbol, orderId=self.latest_trade["orderId"])
        return self.latest_trade_info

    def exit_limit(self, symbol: str, price: float, position_amount: float) -> list:
        """
        Considers the current position you have for a given symbol, and closes it given a price.

        Parameters:
        ------------
        symbol: str            Ex. "BTCUSDT", "ETHUSDT"
        price: float           Ex. 0.001, 0.500, 1.000
        position_amount: int   Ex.  1.0, 2000, 153.5

        returns: dict     Ex. see "enter_market" description
        """
        side = "BUY" if position_amount < 0 else "SELL"
        exitLimitParams = {
            'symbol': symbol,
            'side': side,
            'type': "LIMIT",
            'price': price,
            'timeInForce': "GTC",
            'quantity': abs(position_amount)
        }
        self.latest_trade = self.client.futures_create_order(**exitLimitParams)
        self.latest_trade_info = self.client.futures_get_order(symbol=symbol, orderId=self.latest_trade["orderId"])
        return self.latest_trade_info

    def get_last_x_candles(self, symbol: str, start_minutes_ago: int, end_minutes_ago: int = 0,
                           now: float = time.time()) -> pd.DataFrame:
        """
        Provides a method for getting a set of candlestick data without inputting start and end date.

        Parameters:
        -----------
        symbol: str              Ex. "BTCUSDT", "ETHUSDT"
        start_minutes_ago: int   Ex. 1, 5, 1000
        end_minutes_ago: int     Ex. 1, 5, 1000

        returns pd.DataFrame of candlestick data.
        """
        lst = self.client.futures_klines(symbol=symbol, interval="1m",
                                         startTime=(int(now) - 60 * (start_minutes_ago) - 1) * 1000,
                                         endTime=int(now - 60 * (end_minutes_ago)) * 1000,
                                         limit=abs(start_minutes_ago - end_minutes_ago))
        return into_dataframe(lst)

    def get_necessary_data(self, symbol: str, tf: int) -> pd.DataFrame:
        """
        Gets the necessary data to trade this asset, only a symbol and timeframe need to be inputted

        Parameters:
        symbol: str     Ex. "BTCUSDT", "ETHUSDT"
        tf: int         Ex. 1, 3, 5, 77, 100

        returns, pd.DataFrame of candlestick data

        """
        now = time.time()
        split_number = math.ceil(tf * 235 / 1000) + 1
        ranges = np.ceil(np.linspace(0, tf * 235, num=split_number))
        ranges = [int(i) for i in ranges[::-1]]
        df = pd.DataFrame()

        for i in range(len(ranges)):
            try:
                df = df.append(self.get_last_x_candles(symbol, ranges[i], ranges[i + 1], now))
            except IndexError:
                pass
        return df.drop_duplicates()

    def set_asset(self, symbol: str) -> str:
        """
        Set Symbol of the ticker and load the necessary data with the given timeframe to trade it

        Parameters:
        ------------
        symbol: str     Ex. "BTCUSDT", "ETHUSDT"

        Returns: str response
        """
        self.symbol = symbol
        map_tf = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h", 360: "6h", 480: "8h"}

        startTime = int((time.time() - t.tf * 235 * 60) * 1000)

        if self.tf in [1, 3, 5, 15, 30, 60, 120, 240, 360, 480]:  # 12H, 1D, 3D, 1W, 1M are also recognized

            self.df = into_dataframe(
                self.client.futures_klines(symbol=symbol, interval=map_tf[self.tf], startTime=startTime))
            self.df.drop(self.df.tail(1).index, inplace=True)
        else:
            self.df = t.get_necessary_data(symbol, t.tf)
            load1 = DataLoader()
            self.df = load1.timeframe_setter(self.df, self.tf)

        self.df['Datetime'] = [datetime.fromtimestamp(i / 1000) for i in self.df.index]
        self.df = self.df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
        self.df[["Open", "High", "Low", "Close", "Volume"]] = self.df[
            ["Open", "High", "Low", "Close", "Volume"]].astype(float)
        return f"Symbol changed to {self.symbol}"

    def make_row(self, high=[], low=[], volume=[], count: int = 0, open_price: float = None, open_date: str = None) -> (
    pd.DataFrame, list, list, list, list, list):
        """
        Helper function used to update the last row of a dataframe, considering all of the data in the previous "last row"
        This is used to update the price of the candlestick live, with the uncomplete current candle constantly updating.

        Parameters:
        -----------
        high: list             Ex. [23348, 23350, 23335, 23330, 23339]
        low: list              Ex. [23300, 23345, 23335, 23320, 23300]
        volume: list           Ex. [47, 31, 110, 117, 2, 55]
        count: int             Ex. 1,2,3
        open_price: float      Ex. 23342
        open_date: str         Ex. "2020-08-04 17:33:02"

        returns tuple -> (pd.DataFrame, list, list, list, list, list)
        """
        minutes = 1
        row = self.client.futures_klines(symbol=self.symbol, interval="1m",
                                         startTime=(int(time.time()) - 60 * (minutes) - 1) * 1000)[0]
        timestamp, close = row[0], row[4]
        high.append(float(row[2]))
        low.append(float(row[3]))
        volume.append(float(row[5]))

        if count == 0:
            open_price = row[1]
            open_date = timestamp

        dfrow = pd.DataFrame([[open_date, datetime.fromtimestamp(open_date / 1000), open_price, max(high), min(low),
                               close, sum(volume)]], \
                             columns=["Timestamp", "Datetime", "Open", "High", "Low", "Close", "Volume"])
        dfrow[["Open", "High", "Low", "Close", "Volume"]] = dfrow[["Open", "High", "Low", "Close", "Volume"]].astype(
            float)
        dfrow = dfrow.set_index("Timestamp")
        return dfrow, high, low, volume, open_price, open_date

    def load_asset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sets the trading data to an already existing dataframe, passed in as an argument"""
        self.df = df
        return self.df

    def start_trading(self, strategy: FabStrategy) -> None:
        """
        Starts the process of trading live. Each minute, the last row of the data is updated and Rule #2 is checked,
        otherwise, it waits till the end of the timeframe where it gets the latest data before it checks the rest of the rules.
        If it does see something following the rules, it will buy/short, given the initial parameters it has. (e.g. leverage, quantity)
        This process continues indefinetly, unless interrupted.

        returns None.
        """

        count, open_price = 0, 0
        open_date = None
        high, low, volume = [], [], []

        self.start = True
        strategy.load_data(self.df)
        strategy.create_objects()

        while self.start != False:
            last_date = self.df.iloc[-1]["Datetime"].to_pydatetime()
            current_date = datetime.now()
            diff = (current_date - last_date - timedelta(minutes=self.tf)).seconds / 60

            if round(time.time() % 60, 1) == 0 and diff <= self.tf:

                dfrow, high, low, volume, open_price, open_date = self.make_row(high, low, volume, count, open_price,
                                                                                open_date)
                #                 print(dfrow)

                strategy.load_data(self.df.append(dfrow))
                strategy.create_objects()

                #                 print(f"{self.tf - diff} minutes left")

                if strategy.rule_2_buy_enter(-1, 0.0006) and self.get_position(self.symbol) == 0:
                    trade_info = self.enter_market(self.symbol, "BUY", self.leverage, 2)
                elif strategy.rule_2_short_enter(-1, 0.0006) and self.get_position(self.symbol) == 0:
                    trade_info = self.enter_market(self.symbol, "SELL", self.leverage, 2)

                time.sleep(55)
                count += 1

            elif diff > self.tf:
                self._update_data(math.floor(diff))
                strategy.load_data(self.df)
                strategy.create_objects()

                if strategy.rule_2_short_stop(-1) and self.get_position(self.symbol) < 0 and \
                        self.live_trade_history[-1][-1] == "Rule 2":
                    trade_info = self.exit_market(self.symbol, 2, self.get_position(self.symbol))
                elif strategy.rule_2_buy_stop(-1) and self.get_position(self.symbol) > 0 and \
                        self.live_trade_history[-1][-1] == "Rule 2":
                    trade_info = self.exit_market(self.symbol, 2, self.get_position(self.symbol))
                elif strategy.rule_1_buy_enter(-1) and self.get_position(self.symbol) == 0:
                    trade_info = self.enter_market(self.symbol, "BUY", self.leverage, 1)
                elif strategy.rule_1_buy_exit(-1) and self.get_position(self.symbol) > 0:
                    trade_info = self.exit_market(self.symbol, 1, self.get_position(self.symbol))
                elif strategy.rule_1_short_enter(-1) and self.get_position(self.symbol) == 0:
                    trade_info = self.enter_market(self.symbol, "SELL", self.leverage, 1)
                elif strategy.rule_1_short_exit(-1) and self.get_position(self.symbol) < 0:
                    trade_info = self.exit_market(self.symbol, 1, self.get_position(self.symbol))

                elif strategy.rule_3_buy_enter(-1) and self.get_position(self.symbol) == 0:
                    trade_info = self.enter_market(self.symbol, "BUY", self.leverage, 3)
                elif strategy.rule_3_short_enter(-1) and self.get_position(self.symbol) == 0:
                    trade_info = self.enter_market(self.symbol, "SELL", self.leverage, 3)
                high, low, volume = [], [], []
                count = 0
                time.sleep(1)
#                 print("next")
