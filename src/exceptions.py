
class Exceptions():
    @staticmethod
    def check_trade_status_exists(trade):
        if trade.status != "Enter":
            raise Exception("Trading History doesn't have alternating Entering and Exit positions")

    @staticmethod
    def check_empty_trade_history(trade_history):
        if trade_history == [['List of Trades']]:
            return "Not enough data to provide statistics"

