from datetime import datetime
from os import symlink

import pandas as pd

from dataloader import _DataLoader
from helper import Helper
from trade import Trade
from trading_history import TradeHistory


class TradeExecutor:
    """
    Responsible for the execution of trades on the binance Exchange. The parent class Trader analyzes whether to trade or not.

    Attributes
    ----------
    live_trade_history: [[]]          list of lists - trading data
    trade_journal:      [[]]          pd.DataFrame - dataframe with all raw data from binance response
    account:            str           Response statement from connecting to Binance API
    latest_trade_info   dict          JSON of the API response once trade is placed on the exchange
    latest_trade        dict          JSON of the API response once you fetch the last trade

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
        self.live_trade_history = TradeHistory()
        self.trade_journal = pd.DataFrame()
        self.dic = {"BUY": "Long", "SELL": "Short"}
        self.inverse_dic = dict((y, x) for x,y in self.dic.items())
        self.latest_trade_info = None
        self.latest_trade = None
        self.loader = _DataLoader(db=False)
        self.precisions = {symbol_dic['symbol'] : symbol_dic['quantityPrecision'] for symbol_dic in self.loader.binance.futures_exchange_info()['symbols']}
        
    def how_much_to_buy(self, capital, last_price: str, precision) -> float:
        """Formula that calculates the position size"""
        significant = Helper.sig_fig(capital / float(last_price), 4)
        if significant.is_integer():
            return int(significant)
        result = round(significant, precision)
        return result

    def enter_market(self, client, symbol_side_rule, remaining_capital, number_of_trades=3, reason=None, max_trade_size=None) -> list:
        """
        Creates a order in the exchange, given the symbol

        Parameters:
        ------------
        symbol: str       Ex. "BTCUSDT", "ETHUSDT"
        side: str         Ex. 'BUY', SELL'
        rule_no: int      Ex.  1, 2, 3  

        """

        df_trade_info = pd.DataFrame()
        
        if not symbol_side_rule:
            return 

        list_of_symbols = [symbol for symbol, side, rule in symbol_side_rule]
        print(f"Entering: {', '.join(list_of_symbols)} ")

        for symbol, side, rule in symbol_side_rule:
            last_price = self.loader._get_binance_futures_candles(symbol, tf=1, start_candles_ago=2).iloc[-1, 3]

            per_trade_size = remaining_capital//number_of_trades
            if max_trade_size:
                per_trade_size = max_trade_size if per_trade_size > max_trade_size else per_trade_size  

            enter_market_params = {
                'symbol': symbol,
                'side': self.inverse_dic[side],
                'type': 'MARKET',
                'quantity': self.how_much_to_buy(per_trade_size, last_price, self.precisions[symbol])
            }
            latest_trade = client.futures_create_order(**enter_market_params)
            trade_info = client.futures_get_order(symbol=symbol, orderId=latest_trade["orderId"])
            df_trade_info = df_trade_info.append(pd.DataFrame([trade_info.values()], columns=trade_info.keys()), ignore_index=True)
            df_trade_info = df_trade_info.drop(['price', 'reduceOnly', 'closePosition', 'positionSide', 'stopPrice',
                                                'workingType', 'priceProtect','updateTime', 'origType'], axis=1).set_index('orderId')
            df_trade_info[['enter', 'reason', 'rule']] = [True, reason, rule]
            self.last_trade = df_trade_info
        return df_trade_info

    def exit_market(self, client, positions: dict, reason=None) -> list:
        """
        Considers the current position you have for a given symbol, and closes it accordingly

        Parameters:
        ------------
        symbol: str       Ex. "BTCUSDT", "ETHUSDT"
        rule_no: int      Ex.  1, 2, 3

        Returns: dict     Ex. see "enter_market" description
        """
        df_trade_info = pd.DataFrame()
        list_of_symbols = [symbol for symbol, rest in positions.items()]
        print(f"Exiting: {', '.join(list_of_symbols)} ")
        for symbol, position_amount in positions.items():
            side = "BUY" if position_amount < 0 else "SELL"

            exitMarketParams = {
                'symbol': symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': round(abs(position_amount), self.precisions[symbol])
            }

            latest_trade = client.futures_create_order(**exitMarketParams)
            trade_info = client.futures_get_order(symbol=symbol, orderId=latest_trade["orderId"])
            df_trade_info = df_trade_info.append(pd.DataFrame([trade_info.values()], columns=trade_info.keys()), ignore_index=True)
            df_trade_info = df_trade_info.drop(['price', 'reduceOnly', 'closePosition', 'positionSide', 'stopPrice',
                                    'workingType', 'priceProtect','updateTime', 'origType'], axis=1).set_index('orderId')
            df_trade_info[['enter', 'reason', 'rule']] = [False, reason, None]
            self.last_trade = df_trade_info
        return df_trade_info

    def stop_market(self, client, symbol: str, price: float, position_amount: float) -> list:
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
        self.latest_trade = client.futures_create_order(**stopMarketParams)
        self.latest_trade_info = client.futures_get_order(symbol=symbol, orderId=self.latest_trade["orderId"])

        return self.latest_trade_info

    def enter_limit(self, client, symbol: str, side: str, price: float, capital) -> list:
        """
        Sets a limit order at a given price for a given symbol

        Parameters:
        ------------
        symbol: str       Ex. "BTCUSDT", "ETHUSDT"
        side: str         Ex. 'BUY', SELL'
        price: int        Ex.  10254, 530, 1.01

        Returns: dict     Ex. see "enter_market" desc

        """
        last_price = self.loader._get_binance_futures_candles("BTCUSDT", 2).iloc[-1, 3]

        enterLimitParams = {
            'symbol': symbol,
            'side': side,
            'type': "LIMIT",
            'price': price,
            'timeInForce': "GTC",
            'quantity': self.how_much_to_buy(capital, last_price)
        }

        self.latest_trade = client.futures_create_order(**enterLimitParams)
        self.latest_trade_info = client.futures_get_order(symbol=symbol, orderId=self.latest_trade["orderId"])

        return self.latest_trade_info

    def exit_limit(self, client, symbol: str, price: float, position_amount: float) -> list:
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
        self.latest_trade = client.futures_create_order(**exitLimitParams)
        self.latest_trade_info = client.futures_get_order(symbol=symbol, orderId=self.latest_trade["orderId"])

        return self.latest_trade_info
