import json
import random
import time
from datetime import datetime

import dateparser
import numpy as np
import pandas as pd
import qtrade
from binance.client import Client
from ib_insync import IB, util
from pandas.core import base
from qtrade import Questrade

from helper import Helper
from sql_mapper import SqlMapper


class _DataLoader:
    """
    Private class Responsible for all ETL related tasks. Loads data from csv, fetches data from Binance API.

    Attributes
    -----------
    binance: Client object which is the Binance API Python wrapper

    Methods
    ------------
    _load_csv
    _get_range
    _get_binance_futures_candles
    _timeframe_setter

    Please look at each method for descriptions
    """
    SECOND_TO_MILLISECOND = 1000

    def __init__(self, db=False, qtrade=False, ib=False):
        self.ib = IB()
        self.binance = Client()
        if qtrade:
            self.qtrade = Questrade(token_yaml='C:/Users/haseab/Desktop/Python/PycharmProjects/FAB/local/Workers/access_token.yml', save_yaml=True)
            print('Connected to Questrade API')
        if db:
            self.sql = SqlMapper()
            self.conn = self.sql.connect_psql()
        if ib:
            print(self.ib.connect('127.0.0.1', 7496, 104))

    def _randomly_delete_rows(self, df, percentage_of_data=0.10):
        index_list = []
        for _ in range(len(df)//(1/percentage_of_data)):
            index = random.choice(df.index)
            if index not in index_list:
                index_list.append(index)

        return df.drop(index_list)  

    def _clean_1m_data(self, df):
        start_date = df['timestamp'].iloc[0]
        end_date = df['timestamp'].iloc[-1]

        full_timestamps = pd.DataFrame([time for time in range(start_date, end_date + 60, 60)], columns=['timestamp'])
        full_df = full_timestamps.merge(df.reset_index(), on='timestamp', how='left')
        full_df['volume'] = full_df['volume'].fillna(0.001)
        filled_df = full_df.fillna(method='ffill')
        filled_df['date'] = [datetime.fromtimestamp(timestamp) for timestamp in filled_df['timestamp'].values]
        return filled_df

    def _load_csv_v2(self, csv_url):
        tf = csv_url.split(' ')[2][:-1]
        symbol = csv_url.split(' ')[1]
 
        data = pd.read_csv(csv_url)
        data['timestamp'] = [int(timestamp/1000) for timestamp in data['timestamp']]
        data['date'] = [datetime.fromtimestamp(timestamp) for timestamp in data['timestamp'].values]
        data['tf'] = [tf]*len(data)
        data['symbol'] = [symbol]*len(data)

        data[["open", "high", "low", "close", "volume"]] = data[["open", "high", "low", "close", "volume"]].astype(
            float)

        data = data[['symbol', 'tf', 'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume']]
        return data.set_index(["symbol", "tf", "timestamp"])

    def _load_csv(self, csv_url: str) -> pd.DataFrame:
        """Function used to load 1-minute historical candlestick data with a given csv url
            The important columns are the ones that create the candlestick (open, high, low, close) """
        # Reading CSV File containing 1 min candlestick data
        data = pd.read_csv(csv_url, index_col='timestamp')
        # Converting Timestamp numbers into a new column of readable dates
        data['date'] = [datetime.fromtimestamp(timestamp) for timestamp in data.index]
        data[["open", "high", "low", "close", "volume"]] = data[["open", "high", "low", "close", "volume"]].astype(
            float)
        data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
        return data

    def _get_binance_futures_candles(self, symbol: str, tf: int, start_candles_ago: int, end_candles_ago: int = 0,
                                     now: float = None) -> pd.DataFrame:
        """
        Provides a method for getting a set of candlestick data without inputting start and end date.

        Ex. _get_binance_futures_candles("BTCUSDT", 5, 3) = get candlestick data from 5 minutes ago to 3 minutes ago.

        Parameters:
        -----------
        symbol: str              Ex. "BTCUSDT", "ETHUSDT"
        start_minutes_ago: int   Ex. 1, 5, 1000
        end_minutes_ago: int     Ex. 1, 5, 1000

        :return pd.DataFrame of candlestick data.
        """
        if now == None:
            now = time.time()

        # Defining params to put in exchange API call
        map_tf = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h", 360: "6h", 480: "8h"}
        start_minutes_ago = start_candles_ago*tf
        end_minutes_ago = end_candles_ago*tf

        start_time = Helper().minutes_ago_to_timestamp(start_minutes_ago,now, adjust=self.SECOND_TO_MILLISECOND)
        end_time = Helper().minutes_ago_to_timestamp(end_minutes_ago, now, adjust=self.SECOND_TO_MILLISECOND)
        num_candles = abs(start_candles_ago - end_candles_ago)

        data = self.binance.futures_klines(symbol=symbol, interval=map_tf[tf], startTime=start_time, endTime=end_time, limit=num_candles)

        return Helper.into_dataframe(data, symbol=symbol, tf=tf)
    
    def _get_ibkr_stocks_candles(self, symbol: str, tf: int, start_time, end_time):
        tf_map = {1: "1 min", 5: "5 mins", 15: "15 mins", 30: "30 mins", 
                  60: "1 hour", 240: "4 hours", 1440: "1 day"}
        parsed_start, parsed_end = dateparser.parse(start_time), dateparser.parse(end_time)
        duration = (parsed_end - parsed_start).days + 1

        bars = self.ib.reqHistoricalData(Stock(str(symbol), 'SMART', 'USD'),
                                        endDateTime=parsed_end,
                                        durationStr=f'{duration} D',
                                        barSizeSetting= tf_map[tf],
                                        whatToShow='TRADES',
                                        useRTH=False,
                                        formatDate=1)
        return Helper.into_dataframe(bars, symbol=symbol, tf=tf)

    def _get_range(self, dataframe: pd.DataFrame, start_date: str = None,
                   end_date: str = None) -> pd.DataFrame:
        """Returns the range of 1-min data within specified start & end date from the entire dataset

            Parameters
            ----------
            dataframe: pd.DataFrame object with a Timestamp as its index
            start_date: date in the format of YYYY-MM-DD format
            end_date: date in the format of YYYY-MM-DD format

            :return dataframe
        """
        if start_date == None or end_date == None:
            raise Exception("No Start date given")

        start_date = Helper.string_to_timestamp(start_date)
        end_date = Helper.string_to_timestamp(end_date)

        # Converting from timestamp index to numbered index, then adding numbered index as column
        dataframe_temp = dataframe.reset_index().reset_index().set_index('timestamp')
        start_index = dataframe_temp.loc[start_date, 'index']
        try:
            end_index = dataframe_temp.loc[end_date, 'index']
        except KeyError:
            end_index = dataframe_temp['index'].iloc[-1]

        return dataframe[start_index:end_index+1]

    def _timeframe_setter(self, dataframe: pd.DataFrame, skip: int, shift: int = 0, keep_last_row=False) -> pd.DataFrame:
        """ Vertical way of abstracting data
        Converts minute candlestick data into the timeframe(tf) of choice.
        Parameters
        -----------
        dataframe: the dataframe that is being passed as an argument

        tf: The combination of 1-min candles into one value. Number of 1-min candles combined
                is the timeframe value itself.
                The raw data is in a 1-min timeframe. Dataframe contains the following
                columns: ['open', 'high', 'Low, 'close']. Converting to a X minute timeframe is
                handled differently for every column of the candlestick:

            Close - Since all that matters is the close value every 'tf' minutes, you can skip
                every 'tf' minutes.
                Ex.
                    df['close'] = pd.Series([4.50, 4.60, 4.65, 4.44, 4.21, 4.54, 4.10])
                    _timeframe_setter(df['close'], 2) -> [4.50, 4.65, 4.21, 4.10]
                    _timeframe_setter(df['close'], 3) -> [[4.50, 4.44, 4.10]

            Open - Same rules as Close

            High - Get the maximum 1-min high value given the range of the timeframe
                 Ex.
                     df['close'] = pd.Series([4.50, 4.60, 4.65, 4.44, 4.21, 4.54, 4.10])
                    _timeframe_setter(df['high'], 2) ->  [4.60, 4.65, 4.44, 4.54]
                    _timeframe_setter(df['high'], 3) ->  [4.65, 4.54]

            Low - Same rules as 'high', but instead the minimum of that range

            Volume - Same rules as "High", but instead the sum of that range

        If the range of tf is not even (such as having a tf=2 but only 5 elements), then the
        last value will be dropped

        :return dataframe
        """

        if skip == 1:
            return dataframe
        base_tf = int(dataframe['tf'].iloc[0])
            
        if shift == None:
            # This is making sure that there it shifts so that the last tf candle includes the last 1-minute candle
            shift = skip - len(dataframe) % skip - 1

        dataframe[["open", "high", "low", "close", "volume"]] = dataframe[
            ["open", "high", "low", "close", "volume"]].astype(float)

        # Creating a new dataframe so that the size of the rows of the new dataframe will be the same as the new columns
        df = dataframe.iloc[shift::skip].copy()

        rolled_df = dataframe.rolling(skip)

        high = rolled_df['high'].max()
        low = rolled_df['low'].min()
        volume = rolled_df['volume'].sum()
        close = dataframe.copy()['close']

        # Abstracting based on the highest, lowest and sum respectively.
        df['high'] = np.append(high.iloc[shift+skip::skip].values, high.iloc[-1])
        df['low'] = np.append(low.iloc[shift+skip::skip].values, low.iloc[-1])
        df['volume'] = np.append(volume.iloc[shift+skip::skip].values, volume.iloc[-1])
        # Selecting every nth value in the list, where n is the timeframe
        try:
            df['close'] = close.iloc[shift+skip-1::skip].values
        except ValueError as e:
            df['close'] = np.append(close.iloc[shift+skip-1::skip].values, close.iloc[-1])

        tf = base_tf*skip
        df['tf'] = [tf]*len(df['volume'])
        
        # Dropping the last value, this gets rid of the candle that isn't complete until the end of the tf
        if not keep_last_row:
            df.drop(df.tail(1).index, inplace=True)

        return df.reset_index().set_index(['symbol', 'tf', 'timestamp'])
        
    def _get_fast_questrade_data(self, symbol, start_datetime, end_datetime, tf_str, tf):
        data = self.qtrade.get_historical_data(symbol, start_datetime, end_datetime, tf_str)
        return Helper.into_dataframe(data, symbol=symbol, tf=tf, qtrade=True)

    def _get_fast_ibkr_data(self, symbol, duration, end_datetime, tf_str, tf):
        data = self.ib.reqHistoricalData(Stock(str(symbol), 'SMART', 'USD'),
                                        endDateTime=end_datetime,
                                        durationStr=f'{duration} D',
                                        barSizeSetting= tf_str,
                                        whatToShow='TRADES',
                                        useRTH=False,
                                        formatDate=1)
        return Helper.into_dataframe(data, symbol=symbol, tf=tf)

#################################################   ASYNC FUNCTIONS   ############################################################
    async def _async_get_fast_questrade_data(self, symbol, start_datetime, end_datetime, tf_str, tf):
        data = self.qtrade.get_historical_data(symbol, start_datetime, end_datetime, tf_str)
        return Helper.into_dataframe(data, symbol=symbol, tf=tf, qtrade=True)

    async def _async_get_fast_ibkr_data(self, symbol, duration, end_datetime, tf_str, tf):
        
        data = await self.ib.reqHistoricalDataAsync(Stock(str(symbol), 'SMART', 'USD'),
                                        endDateTime=end_datetime,
                                        durationStr=f'{duration} D',
                                        barSizeSetting= tf_str,
                                        whatToShow='TRADES',
                                        useRTH=False,
                                        formatDate=1)
        print(symbol, tf)
        # return data
        return Helper.into_dataframe(data, symbol=symbol, tf=tf)

##################################################################################################################################

    def get_ibkr_stock_candles(self, symbol, tf, start_time, end_time):
        tf_map = {1: "1 min", 5: "5 mins", 15: "15 mins", 30: "30 mins", 
                  60: "1 hour", 240: "4 hours", 1440: "1 day"}
                  
        start_datetime, end_datetime = dateparser.parse(start_time), dateparser.parse(end_time)
        duration = (end_datetime-start_datetime).days + 1

        data = self.ib.reqHistoricalData(Stock(str(symbol), 'SMART', 'USD'),
                                endDateTime=end_datetime,
                                durationStr=f'{duration} D',
                                barSizeSetting= tf_map[tf], 
                                whatToShow='TRADES',
                                useRTH=False,
                                formatDate=1)
        return util.df(data)
        
    def get_questrade_stock_candles(self, symbol: str, tf: int, start_time, end_time):
        tf_map = {1: "OneMinute", 5: "FiveMinutes", 15: "FifteenMinutes", 30: "HalfHour", 
                  60: "OneHour", 240: "FourHours", 1440: "OneDay"}
        parsed_start, parsed_end = dateparser.parse(start_time), dateparser.parse(end_time)
        parsed_start, parsed_end = parsed_start.strftime('%Y-%m-%d %H:%M:%S.%f'), parsed_end.strftime('%Y-%m-%d %H:%M:%S.%f')
        print('finished converting the times', parsed_start, parsed_end)
        data = self.qtrade.get_historical_data(symbol, parsed_start, parsed_end, tf_map[tf])
        print('got data', len(data))
        return Helper.into_dataframe(data, symbol=symbol, tf=tf, qtrade=True)

    def get_binance_candles(self, symbol, tf, start_date, end_date=None):
        map_tf = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h", 360: "6h", 480: "8h"}
        lst = self.binance.get_historical_klines(symbol=symbol, interval=map_tf[tf], start_str=start_date)
        return Helper.into_dataframe(lst, symbol=symbol, tf=tf)

    def get_all_binance_data(self, symbol, tf, start_date, end_date=None):
        map_tf = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h", 360: "6h", 480: "8h"}
        list_symbol = self.binance.get_historical_klines(symbol=symbol, interval=map_tf[tf], start_str=start_date)
        df_symbol = pd.DataFrame(list_symbol)
        df_symbol.columns = ["timestamp", "open", "high", "low", "close", "volume", "timestamp_end", "", "", "", "", ""]

        ##Fixing Columns
        df_symbol['timestamp'] = [int(timestamp / 1000) for timestamp in df_symbol['timestamp']]
        df_symbol['date'] = [datetime.fromtimestamp(timestamp) for timestamp in df_symbol['timestamp'].values]
        df_symbol['tf'] = [tf[:-1]] * len(df_symbol)
        df_symbol['symbol'] = [symbol] * len(df_symbol)

        df_symbol[["open", "high", "low", "close", "volume"]] = df_symbol[
            ["open", "high", "low", "close", "volume"]].astype(
            float)
        df_symbol = df_symbol[['symbol', 'tf', 'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume']]
        df_symbol = df_symbol.set_index(["symbol", "tf", "timestamp"])

        start_date = str(df_symbol.iloc[0, 0])[:10]

        string = f"Binance {symbol} {tf}m data from {start_date} to {str(datetime.now())[:10]}.csv"
        print(string)
        # df_symbol.to_csv(string)

        return df_symbol
