from dataloader import DataLoader
from illustrator import Illustrator
from trader import Trader
from backtester import Backtester
from analyzer import Analyzer
from binance.client import Client
from fab_strategy import FabStrategy


def sig_fig(x:float, sig:int =2) -> float:
    """ Rounds to the number of significant digits indicated"""
    return round(x, sig - math.ceil(math.log10(abs(x))))


def string_to_timestamp(date:str) -> int:
    """Converts String of form DD-MM-YY into timestamp"""
    return int(time.mktime(datetime.strptime(date, "%d/%m/%Y").timetuple()))*1000


def into_dataframe(lst:list) -> pd.DataFrame:
    """Converts Binance response list into dataframe"""
    return pd.DataFrame(lst,columns = ["Timestamp","Open","High","Low", "Close","Volume","Timestamp_end","","","","",""]).set_index("Timestamp")


if __name__ == "__main__":
    # Instantiating
    t = Trader()
    t.load_account()

    # Setting Initial Conditions

    t.set_timeframe(1)
    t.set_asset('BTCUSDT')

    # Start Trading
    t.start_trading(FabStrategy())