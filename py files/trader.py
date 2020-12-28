from binance.client import Client
import pandas as pd
from dataloader import DataLoader
from datetime import datetime, timedelta
import math
import time
import numpy as np


class Trader():
    def __init__(self):
        self.symbol = None
        self.client = None
        self.capital = None
        self.leverage = 1 / 1000
        self.live_trade_history = ["List of Trades"]
        self.trade_journal = pd.DataFrame()
        self.tf = None
        self.df = None

    def _update_data(self, diff):
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

    def load_account(self):
        info = pd.read_csv(r'C:\Users\owner\Desktop\Python\PycharmProjects\FAB\local\binance_api.txt').set_index('Name')
        API_KEY = info.loc["API_KEY", "Key"]
        SECRET = info.loc["SECRET", "Key"]
        self.client = Client(API_KEY, SECRET)
        self.capital = int(float(self.client.futures_account_balance()[0]['balance'])) + 20000
        return "Welcome Haseab"

    def set_leverage(self, leverage):
        self.leverage = leverage
        return self.leverage

    def set_timeframe(self, tf):
        self.tf = tf
        return self.tf

    def get_position(self, symbol):
        return float([i["positionAmt"] for i in self.client.futures_position_information() if i['symbol'] == symbol][0])

    def enter_market(self, symbol, side, leverage, rule_no):
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

    def exit_market(self, symbol, rule_no, position_amount):
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

    def stop_market(self, symbol, price, position_amount):
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

    def enter_limit(self, symbol, side, price, leverage):
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

    def exit_limit(self, symbol, price, position_amount):
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

    def get_last_x_candles(self, symbol, start_minutes_ago, end_minutes_ago=0, now=time.time()):
        lst = self.client.futures_klines(symbol=symbol, interval="1m",
                                         startTime=(int(now) - 60 * (start_minutes_ago) - 1) * 1000,
                                         endTime=int(now - 60 * (end_minutes_ago)) * 1000,
                                         limit=abs(start_minutes_ago - end_minutes_ago))
        return into_dataframe(lst)

    def get_historical_futures_klines(self, symbol, tf):
        now = time.time()
        split_number = math.ceil(tf * 235 / 1000) + 1
        ranges = np.ceil(np.linspace(0, tf * 235, num=split_number))
        ranges = [int(i) for i in ranges[::-1]]
        lst = pd.DataFrame()

        for i in range(len(ranges)):
            try:
                lst = lst.append(self.get_last_x_candles(symbol, ranges[i], ranges[i + 1], now))
            except IndexError:
                pass
        return lst.drop_duplicates()

    def set_asset(self, symbol):
        """Set Symbol of the ticker"""
        self.symbol = symbol
        map_tf = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h", 360: "6h", 480: "8h"}

        startTime = int((time.time() - t.tf * 235 * 60) * 1000)

        if self.tf in [1, 3, 5, 15, 30, 60, 120, 240, 360, 480]:  # 12H, 1D, 3D, 1W, 1M are also recognized

            self.df = into_dataframe(
                self.client.futures_klines(symbol=symbol, interval=map_tf[self.tf], startTime=startTime))
            self.df.drop(self.df.tail(1).index, inplace=True)
        else:
            self.df = t.get_historical_futures_klines(symbol, t.tf)
            load1 = DataLoader()
            self.df = load1.timeframe_setter(self.df, self.tf)

        self.df['Datetime'] = [datetime.fromtimestamp(i / 1000) for i in self.df.index]
        self.df = self.df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
        self.df[["Open", "High", "Low", "Close", "Volume"]] = self.df[
            ["Open", "High", "Low", "Close", "Volume"]].astype(float)
        return f"Symbol changed to {self.symbol}"

    def make_row(self, high=[], low=[], volume=[], count=0, open_price=None, open_date=None):
        #         row = self.client.get_historical_klines(symbol="BTCUSDT", interval="1m", start_str="61 seconds ago UTC")[0]
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

    def load_asset(self, df):
        self.df = df
        return self.df

    def start_trading(self, strategy):
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
