from binance.client import Client
import pandas as pd

cclass Trader():
    def __init__(self):
        self.symbol = None
        self.client = None
        self.capital = None
        self.tf = 77
        self.df = None

    def load_account(self):
        info = pd.read_csv('binance_api.txt').set_index('Name')
        API_KEY = info.loc["API_KEY","Key"]
        SECRET = info.loc["SECRET","Key"]
        self.client = Client(API_KEY, SECRET)
        self.capital = int(float(self.client.futures_account_balance()[0]['balance'])) + 20000
        return "Welcome Haseab"

    def set_timeframe(self, tf):
        self.tf = tf
        return self.tf

    def set_asset(self, symbol):
        """Set Symbol of the ticker"""
        self.symbol = symbol
        map_tf = {1:"1m",3:"3m", 5:"5m", 15:"15m", 30:"30m", 60:"1h", 120:"2h", 240:"4h", 360:"6h", 480:"8h"}

        start_string = datetime.fromtimestamp(time.time()-self.tf*235*60).strftime("%Y-%m-%d")

        if self.tf in [1,3,5,15,30,60,120,240,360,480]: #12H, 1D, 3D, 1W, 1M are also recognized
            self.df = pd.DataFrame(client.get_historical_klines(symbol=symbol, interval=map_tf[self.tf], start_str = start_string), \
                               columns = ["Timestamp","Open","High","Low", "Close","Volume","Timestamp_end","","","","",""])
            self.df = self.df.reset_index(drop=True).set_index("Timestamp")
        else:
            self.df = pd.DataFrame(client.get_historical_klines(symbol=symbol, interval="1m", start_str = start_string), \
                               columns = ["Timestamp","Open","High","Low", "Close","Volume","Timestamp_end","","","","",""])
            load1 = DataLoader()
            self.df = load1.timeframe_setter(self.df, self.tf).reset_index(drop=True).set_index("Timestamp")

        self.df['Datetime'] = np.array([datetime.fromtimestamp(i/1000) for i in self.df.index])
        self.df = self.df[["Datetime", "Open","High","Low", "Close","Volume"]]
        return f"Symbol changed to {self.symbol}"

    def start_trading(self):
        pass
#         temp, row = [], [None,None,None,None,None]
#         count = 0
#         self.start = True
#         while self.start != False:
#             if round(time.time()%60,1) == 0 and row:
#                 row = client.get_historical_klines(symbol="BTCUSDT", interval="1m", start_str="150 seconds ago UTC")[-1]
#                 dfrow = pd.DataFrame([[row[0],datetime.fromtimestamp(row[0]/1000), row[1],row[2],row[3],row[4],row[5]]], \
#                                      columns = ["Timestamp","Datetime", "Open","High","Low", "Close","Volume"])
#                 dfrow = dfrow.set_index("Timestamp")
#                 run rule 2
#                 count +=1

#             if count > self.tf:
#                 if (df.tail(-1).reset_index() == dfrow.reset_index()).all().all() == False:
#                     df.append(dfrow)
#                 run rule 3
#                 count = 0
