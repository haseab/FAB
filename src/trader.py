from binance.client import Client
import pandas as pd
from datetime import datetime
from helper import Helper
from fab_strategy import FabStrategy
from dataloader import _DataLoader
from trading_history import TradeHistory
import math
import time
import numpy as np
from decouple import config
import os
from illustrator import Illustrator
from ftx import FtxClient
from IPython.display import display, clear_output
from trade_executor import TradeExecutor


class Trader():
    """
    Purpose is to trade live on the Binance Crypto Exchange

    Attributes
    -----------
    symbol: str
    client: Client object - From binance.py
    capital: float - represents how much money is in account
    leverage: float - how many multiples of your money you want to trade
    tf: int - timeframe
    df: pd.DataFrame - data that will be analyzed.
    Methods
    ----------
    load_account
    set_leverage
    set_timeframe
    get_position
    get_necessary_data
    set_asset
    make_row
    load_existing_asset
    start_trading

    Please look at each method for descriptions
    """
    max_candle_history = 231

    def __init__(self, db=True):
        self.start = False
        self.binance = Client()
        self.ftx = FtxClient()
        self.capital = None
        self.leverage = 1 / 1000
        self.tf = None
        self.executor = TradeExecutor()
        self.symbol = None
        self.trade_metrics = None
        self.loader = _DataLoader(db=db)
        self.illustrator = Illustrator()
        self.strategy = FabStrategy()
        self.df = None

    def check_on_switch(self) -> bool:
        """
        Reads a txt file to see whether user wants trading to be on or off
        :return boolean of True or False. True means trading is on, and false means it is off.
        """
        local_path = os.path.dirname(os.path.dirname(os.path.abspath("__file__"))) + r"\local\trade_status.txt"
        status = pd.read_csv(local_path).set_index('index')
        self.start = bool(status.loc[0, 'status'])
        return self.start

    def load_account(self, additional_balance: int = 0) -> str:
        """
        Sign in to account using API_KEY and using Binance API
        """
        # local_path_binance = os.path.dirname(os.path.dirname(os.path.abspath("__file__"))) + r"\local\binance_api.txt"
        # local_path_ftx = os.path.dirname(os.path.dirname(os.path.abspath("__file__"))) + r"\local\ftx_api.txt"
        # info_binance = pd.read_csv(local_path_binance).set_index('Name')
        # info_ftx = pd.read_csv(local_path_ftx).set_index('Name')

        API_KEY_BINANCE = config('API_KEY_BINANCE')
        API_SECRET_BINANCE  = config('API_SECRET_BINANCE')

        API_KEY_FTX = config('API_KEY_FTX')
        API_SECRET_FTX  = config('API_SECRET_FTX')

        self.binance = Client(api_key=API_KEY_BINANCE, api_secret=API_SECRET_BINANCE)
        self.ftx = FtxClient(api_key=API_KEY_FTX, api_secret=API_SECRET_FTX)
        return "Connected to Both Binance and FTX"

    def get_capital(self, additional_balance=0, binance_only=False, ):

        asset_prices = {symbol['symbol'][:-4]: float(symbol['lastPrice']) for symbol in self.binance.get_ticker() if symbol['symbol'][-4:] == 'USDT'}

        # Binance  account balance
        binance_account = self.binance.futures_account()
        binance_futures_balance = float(binance_account['totalCrossWalletBalance']) # + float(binance_account['totalCrossUnPnl'])
        binance_asset_balances = {symbol['asset']: (float(symbol['free']) + float(symbol['locked'])) for symbol in self.binance.get_account()['balances']}
        binance_usdt_balance = binance_asset_balances["USDT"]

        binance_spot_balance = 0
        for binance_asset in binance_asset_balances:
            if binance_asset in asset_prices:
                binance_spot_balance += asset_prices[binance_asset] * binance_asset_balances[binance_asset]

        binance_balance = binance_spot_balance + binance_usdt_balance + binance_futures_balance

        if binance_only:
            self.capital = binance_balance + additional_balance
            return self.capital
        # FTX  account balanced
        ftx_account = self.ftx.get_account_info()
        ftx_balance = ftx_account['totalAccountValue']

        # ftx_asset_balances = {symbol['coin']:symbol['total'] for symbol in self.ftx.get_balances()}
        # ftx_usdt_balance = ftx_asset_balances['USDT'] + ftx_asset_balances['USD']
        #
        # ftx_spot_balance = 0
        # for ftx_asset in ftx_asset_balances:
        #     if ftx_asset in asset_prices:
        #         ftx_spot_balance += asset_prices[ftx_asset] * ftx_asset_balances[ftx_asset]
        # ftx_balance = ftx_spot_balance + ftx_usdt_balance

        self.capital = ftx_balance + binance_balance + additional_balance
        return self.capital

    def set_leverage(self, leverage: float) -> float:
        """Sets the current leverage of the account: should normally be 1. And 0.001 for testing purposes"""
        self.leverage = leverage
        return self.leverage

    def set_timeframe(self, tf: int) -> int:
        """Sets the timeframe of the trading data"""
        self.tf = tf
        return self.tf

    def show_live_chart(self, symbol, tf, refresh_rate=1):
        while True:
            time.sleep(refresh_rate)
            clear_output(wait=True)
            self.show_current_chart('BTCUSDT', tf)

    def show_current_chart(self, symbol=None, tf=None, metric_id=None):
        if metric_id:
            cursor = self.loader.conn.cursor()
            df_metrics = self.loader.sql.SELECT(f"* FROM metrics where metric_id = {metric_id}", cursor)[['symbol', 'tf']]
            symbol = df_metrics['symbol'].iloc[0]
            tf = df_metrics['tf'].iloc[0]

        data = self.set_asset(symbol, tf, max_candles_needed=375, drop_last_row=False)
        return self.illustrator.graph_df(data)

    def get_necessary_data(self, symbol: str, tf: int, max_candles_needed: int) -> pd.DataFrame:
        """
        Gets the minimum necessary data to trade this asset.

        Note: This method is used as a way to tackle the 1000 candle limit that is currently on the Binance API.
        A discrete set of ~1000 group candles will be determined, and then the data will be extracted from each,
        using the _get_binance_futures_candles method, and then all of them will be merged together.

        Parameters:
        symbol: str             - Symbol of price ticker   Ex. "BTCUSDT", "ETHUSDT"
        tf: int                 - Timeframe wanted   Ex. 1, 3, 5, 77, 100
        max_candles_needed: int - Maximum candles needed in the desired timeframe     Ex. 231, 770, 1440

        :return pd.DataFrame of candlestick data

        """
        now = time.time()
        df = pd.DataFrame()
        ranges = Helper.determine_candle_positions(max_candles_needed, tf)


        for i in range(len(ranges)):
            try:
                df = df.append(self.loader._get_binance_futures_candles(symbol, int(ranges[i]), int(ranges[i + 1]), now))
            except IndexError:
                pass
        df['symbol'] = [symbol for _ in range(len(df))]
        return df.drop_duplicates()

    def _update_data(self, diff: int, symbol: str, ) -> None:
        """
        Used to update the trading data with the exchange data so that it is real time

        Parameters:
        -----------
        diff: a number that explains how many minutes of disparity there is between current data and live data.

        :return dataframe with the most up-to-date data.
        """
        minutes_disparity = math.floor(diff) + 1

        # Getting minute candlestick data. Number of minute candlesticks represented by "minutes" variable
        last_few_candles = self.loader._get_binance_futures_candles(symbol, minutes_disparity)
        # Adding a legible Datetime column and using the timestamp data to obtain datetime data
        last_few_candles['datetime'] = Helper.millisecond_timestamp_to_datetime(last_few_candles.index)
        last_few_candles = last_few_candles[["datetime", "open", "high", "low", "close", "volume"]]

        # Abstracting minute data into appropriate tf
        last_few_candles = self.loader._timeframe_setter(last_few_candles, self.tf, 0)

        # Updating original data with new data.
        self.df = self.df.append(last_few_candles).drop_duplicates()

    def get_positions_amount(self, list_of_symbols: [str]) -> [float]:
        """
        Gets the total amount of current position for a given symbol
        Parameters:
        ------------
        symbol: str
            Ex. "BTCUSDT"
                "ETHUSDT"

        Returns float
            Ex. If a 1.000 BTC position is open, it will return 1.0
        """
        position_values = {}
        for symbol in list_of_symbols:
            for coin_futures_info in self.binance.futures_position_information():
                if coin_futures_info['symbol'] == symbol:
                    position_values[symbol] = float(coin_futures_info["positionAmt"])
        return position_values

    def set_asset(self, symbol: str, tf: int, max_candles_needed: int = 231, drop_last_row=False) -> pd.DataFrame:
        """
        Set Symbol of the ticker and load the necessary data with the given timeframe to trade it

        Parameters:
        ------------
        symbol: str     Ex. "BTCUSDT", "ETHUSDT"

        :return str response
        """
        # For Binance API purposes, 240 min needs to be inputted as "4h" on binance when fetching data
        map_tf = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h", 360: "6h", 480: "8h"}
        minutes_ago = tf * (max_candles_needed + 3)
        start_time = Helper().minutes_ago_to_timestamp(minutes_ago, time.time(), 1000)

        if tf in map_tf.keys():  # Note: 12H, 1D, 3D, 1W, 1M are also recognized
            # Fetching data from Binance if it matches the eligible timeframe, as it will be faster
            df = Helper.into_dataframe(
                self.binance.futures_klines(symbol=symbol, interval=map_tf[tf], startTime=start_time))
            if drop_last_row:
                df.drop(df.tail(1).index, inplace=True)
        else:
            # If it doesn't match Binance available timeframes, it must be transformed after fetching 1m data.
            df = self.get_necessary_data(symbol, tf, max_candles_needed)
            df = self.loader._timeframe_setter(df, tf, drop_last_row=drop_last_row)

        # Adding Datetime column for readability of the timestamp. Also more formatting done
        df = df[["date", "open", "high", "low", "close", "volume"]]

        self.df = df
        self.tf = tf
        self.symbol = symbol
        return df

    def get_current_tf_candle(self, tf):
        minute_candles = self.loader._get_binance_futures_candles("BTCUSDT", tf)
        minute_candles['Datetime'] = Helper.millisecond_timestamp_to_datetime(minute_candles.index)
        minute_candles = minute_candles[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

        tf_candle = self.loader._timeframe_setter(minute_candles, tf, drop_last_row=False)
        return tf_candle

    def load__existing_asset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sets the trading data to an already existing dataframe, passed in as an argument"""
        self.df = df
        return self.df

    def check_rule_1(self, client, strategy, executor, position_amount):
        if strategy.rule_1_buy_enter(-1) and position_amount == 0:
            trade_info = executor.enter_market(client, self.symbol, "BUY", self.capital, self.leverage, 1)
        elif strategy.rule_1_buy_exit(-1) and position_amount > 0:
            trade_info = executor.exit_market(client, self.symbol, 1, position_amount)
        elif strategy.rule_1_short_enter(-1) and position_amount == 0:
            trade_info = executor.enter_market(client, self.symbol, "SELL", self.capital, self.leverage, 1)
        elif strategy.rule_1_short_exit(-1) and position_amount < 0:
            trade_info = executor.exit_market(client, self.symbol, 1, position_amount)

    def check_rule_2(self, client, strategy, executor, sensitivity, position_amount):
        if strategy.rule_2_buy_enter(-1, sensitivity) and position_amount == 0:
            trade_info = executor.enter_market(client, self.symbol, "BUY", self.capital, self.leverage, 2)
        elif strategy.rule_2_short_enter(-1, sensitivity) and position_amount == 0:
            trade_info = executor.enter_market(client, self.symbol, "SELL", self.capital, self.leverage, 2)
        elif strategy.rule_2_short_stop(-1) and position_amount < 0 and \
                executor.live_trade_history.last_trade().rule == 2:
            trade_info = executor.exit_market(client, self.symbol, 2, position_amount)
        elif strategy.rule_2_buy_stop(-1) and position_amount > 0 and \
                executor.live_trade_history.last_trade().rule == 2:
            trade_info = executor.exit_market(client, self.symbol, 2, position_amount)

    def check_rule_3(self, client, strategy, executor, position_amount):
        if strategy.rule_3_buy_enter(-1) and position_amount == 0:
            trade_info = executor.enter_market(client, self.symbol, "BUY", self.capital, self.leverage, 3)
        elif strategy.rule_1_buy_exit(-1) and position_amount > 0:
            trade_info = executor.exit_market(client, self.symbol, 1, position_amount)
        elif strategy.rule_3_short_enter(-1) and position_amount == 0:
            trade_info = executor.enter_market(client, self.symbol, "SELL", self.capital, self.leverage, 3)
        elif strategy.rule_1_short_exit(-1) and position_amount < 0:
            trade_info = executor.exit_market(client, self.symbol, 1, position_amount)

    def start_trading(self, strategy: FabStrategy, executor, tf: int,  sensitivity=0, debug=False) -> str:
        """
        Starts the process of trading live. Each minute, the last row of the data is updated and Rule #2 is checked,
        otherwise, it waits till the end of the timeframe where it gets the latest data before it checks the rest of the rules.
        If it does see something following the rules, it will buy/short, given the initial parameters it has. (e.g. leverage, quantity)
        This process continues indefinetly, unless interrupted.

        Returns None.
        """

        raise Exception("Need to review method before running ")

        client = self.binance
        strategy.load_data(self.df)
        strategy.update_moving_averages()

        self.check_on_switch()

        while self.start != False:
            # getting minute candle disparity from real time data
            minute_disparity = Helper.calculate_minute_disparity(self.df, tf)

            if Helper.change_in_clock_minute() and minute_disparity <= tf:
                df_row = self.get_current_tf_candle(minute_disparity)
                position_amount = self.get_position_amount(self.symbol)
                strategy.load_data(self.df.append(df_row))
                strategy.update_moving_averages()

                # Checking only Rule 2, because it needs a minute by minute check.
                self.check_rule_2(client, strategy, executor, sensitivity, position_amount)

                time.sleep(50)

            elif minute_disparity > tf:
                # Updating data using Binance API instead of appending the completed final row from the above code
                self._update_data(math.floor(minute_disparity), self.symbol)
                position_amount = self.get_position_amount(self.symbol)
                strategy.load_data(self.df)
                strategy.update_moving_averages()

                self.check_rule_1(client, strategy, executor, position_amount)
                self.check_rule_3(client, strategy, executor, position_amount)

                time.sleep(1)

            self.check_on_switch()

        return "Trading Stopped"

    def monitor_trades(self) -> pd.DataFrame:
        positions = self.binance.futures_position_information()
        pnl = 1
        df_progress = pd.DataFrame()
        for symbol in positions:
            position_size = float(symbol['positionAmt'])
            if position_size != 0:
                ticker = symbol['symbol']
                side = 'BUY' if position_size > 0 else 'SELL'
                enter_price = float(symbol['entryPrice'])
                last_price = float(symbol['markPrice'])
                if side == 'SELL':
                    pnl = Helper.calculate_short_profitability(enter_price, last_price, 1)
                elif side == 'BUY':
                    pnl = Helper.calculate_long_profitability(enter_price, last_price, 1)
                usd_size = abs(round(position_size * enter_price, 2))
                df_progress = df_progress.append(pd.DataFrame([[ticker, side, enter_price, abs(position_size), usd_size, last_price, pnl]],
                                                              columns = ['symbol', 'side', 'enter_price', 'size', 'usd size', 'current_price', 'pnl']),
                                                 ignore_index=True)
        return df_progress

    def profit_optimization(self, signals, trade_progress_partial, screener):
        trades_to_close, trades_to_enter = [], []

        if len(trade_progress_partial) == 0:
            return [], []

        trade_metrics = self.trade_metrics if type(self.trade_metrics) != type(None) else pd.read_csv("Current metrics for current_trades.csv").set_index('symbol')
        trade_metrics['date'] = trade_metrics['date'].astype('datetime64')
        trade_progress = trade_progress_partial.merge(trade_metrics.reset_index(), on='symbol').set_index('symbol')
        trade_progress['delta profit'] = trade_progress['peak_unrealized_profit'] - trade_progress['pnl']
        trade_progress['candles left'] = ((trade_progress['due date'] - datetime.now()).dt.total_seconds() // trade_progress['tf']).astype('int64')
        potentials = signals['peak_unrealized_profit'].sort_values(ascending=False)

        if len(signals) == 0:
            trades_to_close = self.check_trade_close(screener, trade_progress, trades_to_close)

        trades_to_close, trades_to_enter = self.check_other_signals(potentials, signals, trade_progress, trades_to_close, trades_to_enter)

        self.new = trade_metrics

        for index in trade_metrics.index:
            if index in trades_to_close:
                trade_metrics.drop(index, inplace=True)
        for index in potentials.index:
            if index[0] in trades_to_enter:
                trade_metrics = trade_metrics.append(potentials.loc[index, :])
        print(trades_to_close, trades_to_enter)
        display(trade_metrics)

        trade_metrics.to_csv("Current metrics for current_trades.csv")

        return trades_to_close, trades_to_enter

    def check_other_signals(self, potentials, signals, trade_progress, trades_to_close, trades_to_enter):


        for symbol, tf in potentials.index:
            if len(trade_progress['delta profit']) == 0:
                break

            if potentials.loc[(symbol, tf)] - 1 > min(trade_progress['delta profit']) + 0.0075:
                side = signals.loc[(symbol, tf), 'signal'][1]
                index_of_trade = trade_progress['delta profit'].idxmin()

                trades_to_close.append(index_of_trade)
                trade_progress.drop(index_of_trade, inplace=True)
                trades_to_enter.append((symbol, side))
        return trades_to_close, trades_to_enter

    def score_calculator(self, df_current, df_length, date=False, gain=False):
        weight_dic = {}
        if date:
            weight_dic_x = np.round(np.linspace(0, 100, 1001), 1)
            weight_dic_y = np.round(1.03 ** (weight_dic_x - 80), 5)
            weight_dic = dict(zip(weight_dic_x, weight_dic_y))

        elif gain:
            weight_dic_neg_x = np.round(np.linspace(-50, 0.1, 500), 1)
            weight_dic_neg_y = np.round(15/(weight_dic_neg_x**2 + 15), 5)
            weight_dic_pos_x = np.round(np.linspace(0, 100, 1001), 1)
            weight_dic_pos_y = np.round(0.985**weight_dic_pos_x, 5)
            weight_dic_x = np.append(weight_dic_neg_x, weight_dic_pos_x)
            weight_dic_y = np.append(weight_dic_neg_y, weight_dic_pos_y)
            weight_dic = dict(zip(weight_dic_x, weight_dic_y))

        if not weight_dic:
            raise Exception("date and gain are both set to False. Set one to true")

        relative_current = np.round((1-df_current/df_length)*100, 1)
        score = pd.Series([weight_dic[value] for value in relative_current])
        return score * df_current

    def check_trade_close(self, screener, trade_progress, trades_to_close, max_candle_history=10):
        for symbol in trade_progress.index:
            df = self.set_asset(symbol, trade_progress.loc[symbol, 'tf'], 231 + max_candle_history)
            close_signal, *other = screener.check_for_signals(df, self.strategy,
                                                              max_candle_history=max_candle_history, exit=True)
            # print(close_signal)
            if close_signal:
                trades_to_close.append(symbol)
                continue
            tf = trade_progress.loc[symbol, 'tf']
            elapsed_candles = int(
                (datetime.now() - trade_progress['date'].loc[symbol]).total_seconds() // (60 * tf) + 1)

            if elapsed_candles > trade_progress['num_candles_to_peak'].loc[symbol]:
                trades_to_close.append(symbol)
        return trades_to_close

    def monitor_fab(self, screener, capital, executor=None, number_of_trades=3, max_requests=250, leverage=0.001, recency=-1):
        self.load_account()
        df_metrics = screener.get_metrics_table()
        executor = self.executor if not executor else executor
        client = self.binance

        trade_progress_partial = self.monitor_trades()
        total_invested = trade_progress_partial['usd size'].sum() if len(trade_progress_partial)>0 else 0
        remaining_to_invest = capital - total_invested

        if remaining_to_invest > 25:
            self.trade_metrics = screener.top_trades(trader=self, n=number_of_trades, max_requests=max_requests, df_metrics=df_metrics, df=True, recency=recency)
            self.df_orders = executor.enter_market(client, trade_metrics=self.trade_metrics, capital=remaining_to_invest, leverage=leverage)
            self.trade_metrics.to_csv(f"Current metrics for current_trades.csv")
            print("Sleeping 60 seconds...")
            time.sleep(10)
            time.sleep(10)
            time.sleep(10)
            time.sleep(10)
            time.sleep(10)
            time.sleep(10)

        now = datetime.now()
        now = datetime(year=now.year, month=now.month, day=now.day, hour=now.hour, minute=now.minute)
        while True:
            if datetime.now() >= now + pd.Timedelta(1, 'minute'):
                clear_output(wait=True)
                print(datetime.now())
                print("Getting Top New Trades:.... ")
                df_signals = screener.top_trades(trader=self, df_metrics=df_metrics, max_requests=max_requests, df=True, n=number_of_trades)
                print("Checking Current Trade Progress.... ")
                trade_progress = self.monitor_trades()
                if len(df_signals) == 0 and len(trade_progress) == 0:
                    now = datetime.now()
                    now = datetime(year=now.year, month=now.month, day=now.day, hour=now.hour, minute=now.minute)
                    continue
                print()
                print("Optimizing Trades:... ")
                trades_to_close, trades_to_enter = self.profit_optimization(df_signals, trade_progress, screener=screener)

                if trades_to_close:
                    print(f"Decision Made! Exiting {trades_to_close}")
                    self.close_trade_info = executor.exit_market(client, self.get_positions_amount(trades_to_close))
                if trades_to_enter:
                    print(f"Decision Made! Entering {trades_to_enter}")
                    self.enter_trade_info = executor.enter_market(client, trades_to_enter, capital, leverage=leverage)
                else:
                    print("No Trades made")

                now = datetime.now()
                now = datetime(year=now.year, month=now.month, day=now.day, hour=now.hour, minute=now.minute)
                print()
            print('waiting for next minute...', end='\r')
            time.sleep(0.3)
            print('waiting for next minute.. ', end='\r')
            time.sleep(0.3)
            print('waiting for next minute.  ', end='\r')
            time.sleep(0.3)
            print('waiting for next minute   ', end='\r')
            time.sleep(0.3)
            print('waiting for next minute.  ', end='\r')
            time.sleep(0.3)
            print('waiting for next minute..  ', end='\r')
            time.sleep(0.3)
