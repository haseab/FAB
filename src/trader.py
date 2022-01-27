
import time
from datetime import datetime

import numpy as np
import pandas as pd
from binance.client import Client
from decouple import config
# from ftx import FtxClient
from IPython.display import Markdown, clear_output, display

from dataloader import _DataLoader
from fab_strategy import FabStrategy
from helper import Helper
from illustrator import Illustrator
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

    def __init__(self, binance=True, qtrade=False, db=False, ib=False):
        if binance:
            self.binance = Client()
        # self.ftx = FtxClient()
        self.capital = None
        self.leverage = 1
        self.tf = None
        self.executor = TradeExecutor()
        self.loader = _DataLoader(db=db, qtrade=qtrade, ib=ib)
        self.illustrator = Illustrator()
        self.strategy = FabStrategy()
        self.order_history = pd.DataFrame()
        self.trade_replay = []
        self.precisions = {symbol_dic['symbol'] : symbol_dic['quantityPrecision'] for symbol_dic in self.loader.binance.futures_exchange_info()['symbols']}

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

        # API_KEY_FTX = config('API_KEY_FTX')
        # API_SECRET_FTX  = config('API_SECRET_FTX')

        self.binance = Client(api_key=API_KEY_BINANCE, api_secret=API_SECRET_BINANCE)
        # self.ftx = FtxClient(api_key=API_KEY_FTX, api_secret=API_SECRET_FTX)
        return "Connected to Binance"

    def get_capital(self, additional_balance=4944.2649865):
        initial_margin = float(self.binance.futures_account()['totalInitialMargin'])
        available_balance = float(self.binance.futures_account()['availableBalance'])
        self.capital = (initial_margin + available_balance + additional_balance) * self.leverage
        return self.capital

    def load_futures_trading_history(self, start_time, end_time=None):
        df = pd.DataFrame()
        ranges = Helper.determine_timestamp_positions(start_time, end_time, limit=1) 

        for index, timestamp in enumerate(ranges):
            try:
                df = df.append(pd.DataFrame(self.binance.futures_account_trades(startTime=ranges[index], endTime=ranges[index+1])))
            except (IndexError) as e:
                pass
        return df.reset_index(drop=True)

    def get_tickers(self):
        tickers = pd.DataFrame(self.binance.futures_ticker())
        tickers = tickers.rename({'lastPrice': 'close', 'openPrice':'open', 'highPrice':'high','lowPrice': 'low', 'openTime':'timestamp'}, axis=1)
        tickers['date'] = Helper.millisecond_timestamp_to_datetime(tickers['timestamp'])
        return tickers[['symbol', 'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume']].set_index('symbol')

    def show_live_chart(self, symbol, tf, refresh_rate=1):
        df_tf_sma = self.show_current_chart(symbol, tf)
        while True:
            Helper.sleep(refresh_rate)
            clear_output(wait=True)
            df_tf_sma = self.show_current_chart(symbol, tf, data=df_tf_sma)

    def show_current_chart(self, symbol=None, tf=None, data=()):
        if len(data)==0:
            data = self.set_asset(symbol, tf, max_candles_needed=375, futures=True)
        else:
            row = self.set_asset(symbol, tf, max_candles_needed=1, futures=True)
            if data['date'].iloc[-1] == row['date'].iloc[-1]:
                data = data[:-1].append(row)
            else:
                data = data.append(row)

        tf_data = data['tf'].iloc[0]
        df_tf = self.loader._timeframe_setter(data, tf//tf_data, keep_last_row=True) if tf != tf_data else data
        return self.illustrator.graph_df(df_tf, sma=False)

    def get_necessary_data(self, symbol: str, tf: int, max_candles_needed:int = 235) -> pd.DataFrame:
        df = pd.DataFrame()
        start_datetime = str(Helper.datetime_from_tf(tf, daily_1m_candles=1440, max_candles_needed=max_candles_needed)[0])
        df = df.append(self.loader.get_binance_candles(symbol, tf, start_date=start_datetime))
        df['symbol'] = [symbol for _ in range(len(df))]
        return df.drop_duplicates()

    def get_necessary_data_futures(self, symbol: str, tf: int, max_candles_needed:int = 245) -> pd.DataFrame:
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
                df = df.append(self.loader._get_binance_futures_candles(symbol, tf, int(ranges[i]), int(ranges[i + 1]), now))
            except IndexError:
                pass
        df['symbol'] = [symbol for _ in range(len(df))]
        return df.drop_duplicates()

    def get_single_asset_signals(self, screener, symbol, tf, enter=True, shift=0):
        df = self.set_asset(symbol, tf, max_candles_needed=245)
        binance_df = int(df['tf'].iloc[0])
        df_tf = self.loader._timeframe_setter(df, tf//binance_df, keep_last_row=True, shift=shift)
        strat = FabStrategy()
        df_ma = strat.load_data(df_tf)
        return df_ma, screener._check_for_tf_signals(df_ma, max_candle_history=10, enter=enter)
        
    def set_asset(self, symbol: str, tf: int, max_candles_needed: int = 245, futures=False) -> pd.DataFrame:
        binance_tf = Helper.find_greatest_divisible_timeframe(tf)
        new_max_candles = max_candles_needed*(tf//binance_tf)
        if futures:
            df = self.get_necessary_data_futures(symbol, binance_tf, new_max_candles)
            return df
        df = self.get_necessary_data(symbol, binance_tf, new_max_candles)
        self.df, self.tf, self.symbol= df, tf, symbol
        return df
    def get_current_trade_progress(self, key=()) -> pd.DataFrame:
        if len(key)==0:
            key = self.key
            
        df = pd.DataFrame(self.binance.futures_position_information())
        df[['positionAmt', 'entryPrice', 'markPrice']] = df[['positionAmt', 'entryPrice', 'markPrice']].astype(float)
        df = df[df['positionAmt'] != 0][['symbol', 'positionAmt', 'entryPrice', 'markPrice']].reset_index(drop=True)
        df = df.rename({'positionAmt':'size', 'entryPrice':'enter price', 'markPrice':'current price'}, axis=1)
        df['side'] = np.where(df['size']>0,'BUY', 'SELL')
        df['size'] = df['size'].abs()
        df['usd size'] = df['size']*df['enter price']

        short_profitability = ((0.9996 ** 2) * (2 - df['current price'] / df['enter price']) - 1)*100
        long_profitability = ((0.9996 ** 2) * (df['current price'] / df['enter price']) - 1)*100

        df['pnl (%)'] = np.where(df['side']=='SELL', short_profitability,long_profitability)
        df['pnl (USD)'] = df['usd size']*(df['pnl (%)']/100)
        df['share'] = df['usd size'] / df['usd size'].sum()
        df = df.round(2)
        df = df[['symbol', 'side', 'enter price', 'size', 'usd size', 'current price', 'pnl (%)', 'pnl (USD)', 'share']]

        current_positions = df

        if type(key) != type(None):
            recent_signals = pd.read_csv(r"C:\Users\haseab\Desktop\Python\PycharmProjects\FAB\local\Workers\Recent signals.txt")
            current_positions['tf'] = [int(recent_signals.set_index('symbol').loc[symbol, 'tf']) for symbol in current_positions['symbol'].values]
            current_positions = self.key[['win loss rate', 'avg pnl']].reset_index().merge(current_positions, on=['symbol','tf'])

        self.current_positions = current_positions
        return current_positions

    def _check_trade_close(self, screener, current_positions):
        trades_to_close = []

        for symbol, tf in current_positions.reset_index().set_index(['symbol','tf']).index:
            try:
                df_tf= screener.df_dic[(symbol, tf)]
                df_ma = self.strategy.load_data(df_tf)
                display(Markdown(f"## {symbol} {tf}"))
                display(self.illustrator.graph_df(df_ma[-50:], flat=True))
                enter_signal, enter_x_last_row, *rest= screener._check_for_tf_signals(df_tf, max_candle_history=10, enter=True)
                close_signal, exit_x_last_row, *rest= screener._check_for_tf_signals(df_tf, max_candle_history=10, enter=False)

                self.df_tf, self.close_signal, self.exit_x_last_row = df_tf, close_signal, exit_x_last_row,

                if close_signal:
                    # raise Exception(close_signal, exit_x_last_row, enter_x_last_row)
                    if enter_x_last_row and exit_x_last_row < enter_x_last_row:
                        continue
                    
                    last_entry_date = pd.Timestamp(self.recent_signals.reset_index().set_index(['symbol','tf']).loc[(symbol,tf), 'date'])
                    last_exit_date = self.df['date'].iloc[self.exit_x_last_row]
                    if last_entry_date > last_exit_date:
                        print('failed')
                        continue

                    trades_to_close.append(symbol)
                    print(f'{symbol} Trade Closed')
            except AttributeError as e:
                return []
                
        return trades_to_close

    def trade_free_capital(self, executor, current_positions, remaining_to_invest, trades_to_enter):
        print("-----------------------------------------------")
        print(f"Free capital: {remaining_to_invest} USD")
        print("-----------------------------------------------", end="\n\n\n")
        final_trades_to_enter = self.check_against_max_trade_size(trades_to_enter, current_positions, self.max_trade_size)
        self.order_history = self.order_history.append(executor.enter_market(client=self.binance, symbol_side_rule=final_trades_to_enter, remaining_capital=remaining_to_invest, 
                                                                             number_of_trades=len(trades_to_enter), reason='free_capital', max_trade_size = self.max_trade_size))
        display(self.order_history.tail(len(trades_to_enter)))
        self.order_history.to_csv("FAB Order History.csv")
        self.trade_replay.append(self.get_current_trade_progress(printout=False))

    def calculate_remaining_capital(self, current_positions, capital):
        total_invested = current_positions['usd size'].sum() if len(current_positions) > 0 else 0
        remaining_to_invest = capital - total_invested
        return remaining_to_invest

    def get_positions_amount(self, list_of_symbols: list, full_close=False, divide_by=1, max_trades=3) -> list:
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
        # [{'symbol':'CHRUSDT', 'positionAmt':1}]
        position_values = {}
        for symbol in list_of_symbols:  
            last_price = float(self.binance.futures_mark_price(symbol=symbol)['markPrice'])

            for coin_futures_info in self.binance.futures_position_information():
                if coin_futures_info['symbol'] == symbol:
                    amount = float(coin_futures_info["positionAmt"])
                    amount = round(amount, self.precisions[symbol])
                    if divide_by > 1:
                        amount = round(amount - amount/divide_by, self.precisions[symbol])
                        position_values[symbol] = amount if amount > 0 else 10**(-self.precisions[symbol])
                    if full_close:
                        position_values[symbol] = amount
                    else:
                        return position_values
        return position_values
    
    def get_key(self, essential_metrics, win_loss_threshold = 0.350, pnl_threshold=1.025, data_point_threshold=5):
        key = essential_metrics[essential_metrics['amount of data'] >= data_point_threshold]
        key = key[key['avg pnl'] > pnl_threshold]
        key = key[key['win loss rate'] > win_loss_threshold]
        key['win loss rate'] = key['win loss rate'].round(1)
        key['wnl drawdown'] = 0.4*key['win loss rate']+0.6*key['longest drawdown']
        master_key = key.reset_index().set_index(['symbol', 'tf']).sort_values('wnl drawdown', ascending=False)
        # master_key = key.reset_index().set_index(['symbol', 'tf']).sort_values(by=['win loss rate', 'avg pnl'], ascending=[False, False])
        self.key = master_key
        return master_key

    def _profit_optimization(self, key, current_positions, trades_to_enter, number_of_trades):
        """
        Returns:
            final_trades_to_enter: Ex. [('ZILUSDT', "Short", 'Rule 2'), ('BTCUSDT', 'Long', 'Rule 1'), ("ETHUSDT", "Short", 'Rule 3')]
            trades_to_close:       Ex. ['ADAUSDT']
        """
        print()
        print("Optimizing Trades:... ")
        
        temp_metrics = key.reset_index().set_index(['symbol', 'side', 'rule no'])
        trades_of_interest = [(symbol, int(temp_metrics.loc[(symbol, side, rule), 'tf']))  for symbol, side, rule in trades_to_enter if (symbol, side, rule) in temp_metrics.index]
        
        if not trades_of_interest:
            print("Didn't make it past key filter", end='\n\n')
            return [], []

        current_position_symbols_tf_pair = Helper().nparray_to_tuple(current_positions.reset_index()[['symbol','tf']].values)

        all_symbols = list(set(trades_of_interest).union(set(current_position_symbols_tf_pair)))

        self.current_position_symbols_tf_pair, self.trades_of_interest, self.all_symbols = current_position_symbols_tf_pair, trades_of_interest, all_symbols
    
        self.top_x_symbols = sorted(all_symbols, key=lambda x: (key.loc[x, 'win loss rate'], key.loc[x, 'avg pnl']), reverse=True)[:number_of_trades]

        partial_trades_to_enter = list(set(self.top_x_symbols).intersection(set(trades_of_interest)).difference(set(current_position_symbols_tf_pair)))
        trades_to_close = list(set(current_position_symbols_tf_pair).difference(set(self.top_x_symbols)))
        trades_to_close = [symbol for symbol, tf in trades_to_close] 

        final_trades_to_enter = [(symbol, side, rule) for (symbol, tf), (symbol, side, rule) in zip(trades_of_interest, trades_to_enter) if (symbol,tf) in partial_trades_to_enter]

        for tup in trades_of_interest:
            if tup in final_trades_to_enter:
                raise Exception(tup, trades_of_interest, final_trades_to_enter)

        return trades_to_close, final_trades_to_enter

    def get_dividing_factor(self, current_positions, final_trades_to_enter, number_of_trades):
        dividing_factor = (len(current_positions) + len(final_trades_to_enter))/len(current_positions)
        minimum_threshold = current_positions['usd size'].sum()*0.95 // len(current_positions)
        if dividing_factor > 1 and minimum_threshold > (self.capital)//number_of_trades:
            print(minimum_threshold, (self.capital)//number_of_trades)
            return dividing_factor
        return 1

    def optimize_trades(self, executor, current_positions, trades_to_enter, number_of_trades):
        if len(current_positions) != 0:
            self.position_capital = self.get_capital()
            trades_to_close, final_trades_to_enter = self._profit_optimization(self.key, current_positions, trades_to_enter, number_of_trades)
            
            if trades_to_close:
                self.order_history = self.order_history.append(executor.exit_market(self.binance, self.get_positions_amount(trades_to_close, full_close=True), reason='lower rank'))
                display(self.order_history)
                current_positions = self.get_current_trade_progress(printout=False)

            if final_trades_to_enter:
                dividing_factor = self.get_dividing_factor(current_positions, final_trades_to_enter, number_of_trades)
                positions_amount = self.get_positions_amount(current_positions['symbol'].values, divide_by=dividing_factor, max_trades=number_of_trades)

                if positions_amount:
                    self.order_history = self.order_history.append(executor.exit_market(self.binance, positions_amount, reason='making space'))
                self.position_capital = self.get_capital()
                final_trades_to_enter = self.check_against_max_trade_size(final_trades_to_enter, current_positions, self.max_trade_size)
                self.remaining_capital = self.calculate_remaining_capital(current_positions, self.position_capital)
                self.order_history = self.order_history.append(executor.enter_market(self.binance, final_trades_to_enter, self.remaining_capital, 
                                                                                        len(final_trades_to_enter), reason='higher rank', 
                                                                                        max_trade_size = self.max_trade_size))
                display(self.order_history)
            else:
                print("No Trades made")

        self.trades_to_close, self.final_trades_to_enter = trades_to_close, final_trades_to_enter
        self.order_history.to_csv("FAB Order History.csv")
        self.trade_replay.append(self.get_current_trade_progress(printout=False))

    def close_any_old_trades(self, screener, executor, current_positions):
        if len(current_positions) == 0:
            return False
        trades_to_close = self._check_trade_close(screener, current_positions)
        if trades_to_close:
            self.order_history = self.order_history.append(executor.exit_market(self.binance, self.get_positions_amount(trades_to_close, full_close=True), reason='trade closed'))

    def update_trades_to_enter(self, current_positions, trades_to_enter):
        if len(trades_to_enter) ==0:
            trades_to_enter = [(symbol, self.executor.dic[side], None) for symbol, side in current_positions[['symbol','side']].values]
        return trades_to_enter

    def check_against_max_trade_size(self, trades_to_enter, current_positions, max_trade_size):
        final_trades_to_enter = []
        for symbol, side, rule in trades_to_enter:
            if symbol in current_positions['symbol'].values and current_positions.set_index('symbol').loc[symbol, 'usd size'] >= max_trade_size*0.99: 
                print('worked')
                continue
            final_trades_to_enter.append((symbol, side, rule))
        return final_trades_to_enter

    def monitor_fab(self, screener, df_metrics, tfs, number_of_trades=3, leverage=1, recency=-1, additional_balance=0, max_trade_size=None):
        # self.order_history = pd.DataFrame()
        self.load_account()
        executor = self.executor
        self.leverage = leverage

        essential_metrics = Helper.drop_extra_delays(df_metrics, screener.tf_delay_match)
        self.starting_capital= self.get_capital(additional_balance=additional_balance)
        self.max_trade_size = max_trade_size
        self.key = self.get_key(essential_metrics)
        # self.key.to_csv ('Optimization Function Key.csv')
        
        now = Helper.current_minute_datetime()
        while True:
            if datetime.now() >= now + pd.Timedelta(1, 'minute'):
                clear_output(wait=True)
                print("Current time: \t", datetime.now(), end='\n\n')

                current_positions = self.get_current_trade_progress(key=self.key)
                if len(current_positions) > number_of_trades:
                    raise Exception(f"You have more than {number_of_trades} active trades! Close first")
                if current_positions['usd size'].sum() > self.starting_capital*1.5:
                    raise Exception('We are too overexposed!', current_positions['usd size'].sum())

                self.close_any_old_trades(screener, executor, current_positions)
                
                remaining_capital = self.calculate_remaining_capital(current_positions, self.capital)
                
                ## trades_to_enter is a list of lists (Ex. [('BANDUSDT', 'Short', 'Rule2'), ('BCHUSDT', 'Short', 'Rule 1')]
                ## df_recent_signals is a regular screener dataframe
        
                trades_to_enter, recent_signals = screener.top_trades(trader=self, df_metrics=df_metrics, tfs=tfs, n=number_of_trades, recency=recency)

                self.trades_to_enter, self.recent_signals = trades_to_enter, recent_signals

                print("Signals:")
                display(recent_signals)

                trades_left = number_of_trades - len(current_positions)

                if remaining_capital > 100 and trades_left > 0 and trades_to_enter:
                    self.trade_free_capital(executor, current_positions, remaining_capital, trades_to_enter)

                elif remaining_capital >100 and len(current_positions) != 0:
                    trades_to_enter = self.update_trades_to_enter(current_positions=current_positions, trades_to_enter=trades_to_enter)
                    self.trade_free_capital(executor, current_positions, remaining_capital, trades_to_enter)
                elif not trades_to_enter:
                    Helper.sleep(60)
                    now = Helper.current_minute_datetime()
                    continue
                else:
                    print("Leverage: ", self.leverage)
                    print("Trades to Enter: ", trades_to_enter)
                    print("Max Trades: ", number_of_trades)
                    self.optimize_trades(executor, current_positions, trades_to_enter, number_of_trades)
                Helper.sleep(60)
                now = Helper.current_minute_datetime()
                print()
            Helper.output_loading()



