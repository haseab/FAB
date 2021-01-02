from dataloader import DataLoader
from illustrator import Illustrator
from trader import Trader
from backtester import Backtester
from analyzer import Analyzer
from binance.client import Client
from fab_strategy import FabStrategy



if __name__ == "__main__":
    # Instantiating
    t = Trader()
    t.load_account()

    # Setting Initial Conditions

    t.set_timeframe(1)
    t.set_asset('BTCUSDT')

    # Start Trading
    t.start_trading(FabStrategy())