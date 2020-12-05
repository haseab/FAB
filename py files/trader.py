from binance.client import Client
import pandas as pd

class Trader():
    def __init__(self):
        self.symbol = None
        self.client = None
        self.capital = None

    def load_account(self):
        info = pd.read_csv('binance_api.txt').set_index('Name')
        API_KEY = info.loc["API_KEY", "Key"]
        SECRET = info.loc["SECRET", "Key"]
        self.client = Client(API_KEY, SECRET)
        self.capital = int(float(self.client.futures_account_balance()[0]['balance']))
        return "Welcome Haseab"

    def set_asset(self, symbol):
        """Set Symbol of the ticker"""
        self.symbol = symbol
        return f"Symbol to {self.symbol}"

    def start_trading(self):
        pass
