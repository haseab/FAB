from trader import Trader
from backtester import Backtester
from fab_strategy import FabStrategy



if __name__ == "__main__":

    # If you want to backtest, see below

    # Instantiating
    b = Backtester()
    b.set_asset("BTCUSDT")
    b.set_timeframe(77)
    b.set_date_range("2018-01-01", "2019-01-01")

    results = b.start_backtest

    # If you want to trade live, see below

    # Instantiating
    t = Trader()
    t.load_account()

    # Setting Initial Conditions

    t.set_timeframe(1)
    t.set_leverage(1) # In case you want to trade for real
    t.set_asset('BTCUSDT')

    # Start Trading
    strategy = FabStrategy()
    t.start_trading()