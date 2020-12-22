from dataloader import DataLoader
from illustrator import Illustrator
from trader import Trader
from backtester import Backtester
from analyzer import Analyzer
from binance.client import Client
from fab_strategy import FabStrategy


def sig_fig(self,x, sig=2):
    return round(x, sig - math.ceil(math.log10(abs(x))))


def string_to_timestamp(date):
    return time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple())


if __name__ == "__main__":
    # Instantiating
    t = Trader()
    t.load_account()

    # Setting Initial Conditions

    t.set_timeframe(1)
    t.set_asset('BTCUSDT')

    # Start Trading
    t.start_trading(FabStrategy())