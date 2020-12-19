from binance.client import Client
import pandas as pd
from dataloader import DataLoader
import math


class Trader():
    def __init__(self):
        self.symbol = None
        self.client = None
        self.capital = None
        self.leverage = 1 / 1000
        self.live_trade_history = ["List of Trades"]
        self.trade_journal = pd.DataFrame()
        self.tf = 77
        self.df = None

    def _update_data(self, diff):
        last_price = pd.DataFrame(self.client.get_historical_klines(symbol=symbol, interval="1m",
                                                                    start_str=f"{math.floor(diff) + 1} minutes ago UTC"), \
                                  columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "Timestamp_end", "",
                                           "", "", "", ""]).set_index("Timestamp")
        last_price['Datetime'] = np.array([datetime.fromtimestamp(i / 1000) for i in last_price.index])
        last_price = last_price.append(last_price.tail(1))
        last_price = last_price[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
        load1 = DataLoader()
        last_price = load1.timeframe_setter(last_price, self.tf, 0)
        self.df = self.df.append(last_price).drop_duplicates()

    def load_account(self):
        info = pd.read_csv('binance_api.txt').set_index('Name')
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

    def enter_market(self, symbol, side, leverage, rule_no):
        last_price = float(
            self.client.get_historical_klines(symbol=symbol, interval="1m", start_str="150 seconds ago UTC")[-1][4])
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

    def exit_market(self, symbol, rule_no):
        position_amount = float(
            [i["positionAmt"] for i in t.client.futures_position_information() if i['symbol'] == symbol][0])
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

    def stop_market(self, symbol, price):
        position_amount = float(
            [i["positionAmt"] for i in t.client.futures_position_information() if i['symbol'] == symbol][0])
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

    def exit_limit(self, symbol, price):
        position_amount = float(
            [i["positionAmt"] for i in t.client.futures_position_information() if i['symbol'] == symbol][0])
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

    def set_asset(self, symbol):
        """Set Symbol of the ticker"""
        self.symbol = symbol
        map_tf = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h", 360: "6h", 480: "8h"}

        start_string = datetime.fromtimestamp(time.time() - self.tf * 235 * 60).strftime("%Y-%m-%d")

        if self.tf in [1, 3, 5, 15, 30, 60, 120, 240, 360, 480]:  # 12H, 1D, 3D, 1W, 1M are also recognized
            self.df = pd.DataFrame(
                self.client.get_historical_klines(symbol=symbol, interval=map_tf[self.tf], start_str=start_string), \
                columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "Timestamp_end", "", "", "", "", ""])
            self.df.drop(self.df.tail(1).index, inplace=True)
            self.df = self.df.reset_index(drop=True).set_index("Timestamp")
        else:
            self.df = pd.DataFrame(
                self.client.get_historical_klines(symbol=symbol, interval="1m", start_str=start_string), \
                columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "Timestamp_end", "", "", "", "", ""])
            load1 = DataLoader()
            self.df = load1.timeframe_setter(self.df, self.tf).reset_index(drop=True).set_index("Timestamp")

        self.df['Datetime'] = np.array([datetime.fromtimestamp(i / 1000) for i in self.df.index])
        self.df = self.df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
        self.df[["Open", "High", "Low", "Close", "Volume"]] = self.df[
            ["Open", "High", "Low", "Close", "Volume"]].astype(float)
        return f"Symbol changed to {self.symbol}"

    def make_row(self, high=[], low=[], volume=[], count=0, open_price=None, open_date=None):
        row = self.client.get_historical_klines(symbol="BTCUSDT", interval="1m", start_str="61 seconds ago UTC")[0]
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
            diff = (current_date - last_date - timedelta(minutes=t.tf)).seconds / 60

            if round(time.time() % 60, 1) == 0 and diff <= self.tf:

                dfrow, high, low, volume, open_price, open_date = self.make_row(high, low, volume, count, open_price,
                                                                                open_date)
                #                 print(dfrow)

                strategy.load_data(self.df.append(dfrow))
                strategy.create_objects()

                #                 print(f"{self.tf - diff} minutes left")

                if strategy.rule_2_buy_enter(-1, 0.001) and self.live_trade_history[-1][1] != "Enter":
                    trade_info = self.enter_market(self.symbol, "BUY", self.leverage, 2)
                elif strategy.rule_2_buy_stop(-1) and self.live_trade_history[-1][-1] == "Rule 2" and \
                        self.live_trade_history[-1][:2] == ["Long", 'Enter']:
                    trade_info = self.exit_market(self.symbol, 2)
                elif strategy.rule_2_short_enter(-1, 0.001) and self.live_trade_history[-1][1] != "Enter":
                    trade_info = self.enter_market(self.symbol, "SELL", self.leverage, 2)
                elif strategy.rule_2_short_stop(-1) and self.live_trade_history[-1][:2] == ["Short", 'Enter'] and \
                        self.live_trade_history[-1][-1] == "Rule 2":
                    trade_info = self.exit_market(self.symbol, 2)

                #                 time.sleep(55)
                count += 1

            elif diff > self.tf:
                self._update_data(math.floor(diff))
                if strategy.rule_1_buy_enter(-1) and self.live_trade_history[-1][1] != "Enter":
                    trade_info = self.enter_market(self.symbol, "BUY", self.leverage, 1)
                elif strategy.rule_1_buy_exit(-1) and self.live_trade_history[-1][:2] == ["Long", 'Enter']:
                    trade_info = self.exit_market(self.symbol, 1)
                elif strategy.rule_1_short_enter(-1) and self.live_trade_history[-1][1] != "Enter":
                    trade_info = self.enter_market(self.symbol, "SELL", self.leverage, 1)
                elif strategy.rule_1_short_exit(-1) and self.live_trade_history[-1][:2] == ["Short", 'Enter']:
                    trade_info = self.exit_market(self.symbol, 1)

                elif strategy.rule_3_buy_enter(-1) and self.live_trade_history[-1][1] != "Enter":
                    trade_info = self.enter_market(self.symbol, "BUY", self.leverage, 1)
                elif strategy.rule_3_short_enter(-1) and self.live_trade_history[-1][1] != "Enter":
                    trade_info = self.enter_market(self.symbol, "SELL", self.leverage, 1)
                high, low, volume = [], [], []
                count = 0
#                 print("next")
