from trader import Trader
from helper import Helper
from datetime import datetime
from trade import Trade


class TradeExecutor(Trader):
    """
    Attributes
    ----------
    All attributes from Trader class
    account: str                -> Response statement from connecting to Binance API

    Methods
    -------
    how_much_to_buy
    enter_market
    exit_market
    stop_market
    enter_limit
    exit_limit

    Please look at each method for descriptions
    """

    def __init__(self):
        super().__init__()
        self.account = self.load_account()
        self.helper = Helper()

    def how_much_to_buy(self, last_price):
        """Formula that calculates the position size"""
        return round(Helper.sig_fig(self.capital * self.leverage / float(last_price), 4), 3)

    def enter_market(self, symbol: str, side: str, rule_no: int) -> list:
        """
        Creates a order in the exchange, given the symbol

        Parameters:
        ------------
        symbol: str       Ex. "BTCUSDT", "ETHUSDT"
        side: str         Ex. 'BUY', SELL'
        leverage: float   Ex. 0.001, 0.500, 1.000
        rule_no: int      Ex.  1, 2, 3

        Returns: dict
            Ex. {'orderId': ***********,
                 'symbol': 'BTCUSDT',
                 'status': 'FILLED',
                 'clientOrderId': '**************',
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
        last_price = self.loader._get_binance_futures_candles("BTCUSDT", minutes).iloc[-1, 3]
        #         assert self.how_much_to_buy(last_price) <= self.capital*leverage/last_price + 1

        enterMarketParams = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': self.how_much_to_buy(last_price)
        }

        # Creating order to the exchange
        self.latest_trade = self.client.futures_create_order(**enterMarketParams)
        self.latest_trade_info = self.client.futures_get_order(symbol=symbol, orderId=self.latest_trade["orderId"])

        # Getting Datetime of trade
        date = str(datetime.fromtimestamp(self.latest_trade_info['time'] / 1000))[:19]

        # Creating dictionary to convert "BUY" to "Long", for the sake of maintaining trade_history form.
        dic = {"BUY": "Long", "SELL": "Short"}

        # Note: trade_history takes the form [LONG/SHORT, ENTER/EXIT, DATETIME, PRICE, RULE #]
        self.live_trade_history.append(
            Trade([dic[side], "Enter", date, self.latest_trade_info['avgPrice'][:8], f"Rule {rule_no}"]))

        # Adding Raw order data to a separate list
        self.trade_journal = self.trade_journal.append([self.latest_trade_info])

        return self.latest_trade_info

    def exit_market(self, symbol: str, rule_no: int, position_amount: float) -> list:
        """
        Considers the current position you have for a given symbol, and closes it accordingly

        Parameters:
        ------------
        symbol: str       Ex. "BTCUSDT", "ETHUSDT"
        rule_no: int      Ex.  1, 2, 3

        Returns: dict     Ex. see "enter_market" description
        """
        # Determines whether currently long or short, and it takes the other side (in order to close it)
        side = "BUY" if position_amount < 0 else "SELL"

        exitMarketParams = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': abs(position_amount)
        }

        # Creating order to the exchange
        self.latest_trade = self.client.futures_create_order(**exitMarketParams)
        self.latest_trade_info = self.client.futures_get_order(symbol=symbol, orderId=self.latest_trade["orderId"])

        # Getting the date of the trade
        date = str(datetime.fromtimestamp(self.latest_trade_info['time'] / 1000))[:19]

        # Creating dictionary for the sake of maintaining trade_history form (since its a Short Exit, not a Buy Exit)
        dic = {"BUY": "Short", "SELL": "Long"}

        # Trade history list takes the form [LONG/SHORT, ENTER/EXIT, DATETIME, PRICE, RULE #]
        self.live_trade_history.append(
            Trade([dic[side], "Exit", date, self.latest_trade_info['avgPrice'][:8], f"Rule {rule_no}"]))

        # Adding Raw order data to a separate list
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

        Returns: dict          Ex. see "enter_market" desc

        """
        # Determines whether currently long or short, and it takes the other side (in order to close it)
        side = "BUY" if position_amount < 0 else "SELL"

        stopMarketParams = {
            'symbol': symbol,
            'side': side,
            'type': 'STOP_MARKET',
            'stopPrice': price,
            'quantity': abs(position_amount)
        }

        # Creating order to the exchange
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

        Returns: dict     Ex. see "enter_market" desc

        """
        enterLimitParams = {
            'symbol': symbol,
            'side': side,
            'type': "LIMIT",
            'price': price,
            'timeInForce': "GTC",
            'quantity': round(Helper.sig_fig(self.capital * leverage / (last_price), 4), 3)
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

        Returns: dict     Ex. see "enter_market" description
        """
        # Determines whether currently long or short, and it takes the other side (in order to close it)
        side = "BUY" if position_amount < 0 else "SELL"

        exitLimitParams = {
            'symbol': symbol,
            'side': side,
            'type': "LIMIT",
            'price': price,
            'timeInForce': "GTC",
            'quantity': abs(position_amount)
        }
        # Creating order to the exchange
        self.latest_trade = self.client.futures_create_order(**exitLimitParams)
        self.latest_trade_info = self.client.futures_get_order(symbol=symbol, orderId=self.latest_trade["orderId"])

        return self.latest_trade_info
