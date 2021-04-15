import time
import pandas as pd
from datetime import datetime
from binance.client import Client
from helper import Helper
import sqlalchemy
import json
from sql_mapper import SqlMapper

class _DataLoader:
    """
    Private class Responsible for all ETL related tasks. Loads data from csv, fetches data from Binance API.

    Attributes
    -----------
    client: Client object which is the Binance API Python wrapper

    Methods
    ------------
    _load_csv
    _get_range
    _get_binance_futures_candles
    _timeframe_setter

    Please look at each method for descriptions
    """
    SECOND_TO_MILLISECOND = 1000

    def __init__(self, db=True):
        self.client = Client()
        if db:
            self.sql = SqlMapper()
            self.conn = self.sql.connect_psql()
        # with open("fab_engine.txt", 'r') as file:
        #     self.engine = sqlalchemy.create_engine(file.readline())

    def _load_csv_v2(self, csv_url):
        tf = csv_url.split(' ')[2][:-1]
        symbol = csv_url.split(' ')[1]

        data = pd.read_csv(csv_url)
        data['timestamp'] = [int(timestamp/1000) for timestamp in data['timestamp']]
        data['date'] = [datetime.fromtimestamp(timestamp) for timestamp in data['timestamp'].values]
        data['timeframe'] = [tf]*len(data)
        data['symbol'] = [symbol]*len(data)

        data[["open", "high", "low", "close", "volume"]] = data[["open", "high", "low", "close", "volume"]].astype(
            float)

        data = data[['symbol', 'timeframe', 'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume']]
        return data.set_index(["symbol", "timeframe", "timestamp"])

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

    def _get_binance_futures_candles(self, symbol: str, start_minutes_ago: int, end_minutes_ago: int = 0,
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
        start_time = Helper().minutes_ago_to_timestamp(start_minutes_ago,now, adjust=self.SECOND_TO_MILLISECOND)
        end_time = Helper().minutes_ago_to_timestamp(end_minutes_ago, now, adjust=self.SECOND_TO_MILLISECOND)
        num_candles = abs(start_minutes_ago - end_minutes_ago)

        data = self.client.futures_klines(symbol=symbol, interval="1m", startTime=start_time, endTime=end_time, limit=num_candles)

        return Helper.into_dataframe(data)

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

    def _timeframe_setter(self, dataframe: pd.DataFrame, tf: int, shift: int = 0, drop_last_row=True) -> pd.DataFrame:
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
        if shift == None:
            # This is making sure that there it shifts so that the last tf candle includes the last 1-minute candle
            shift = tf - len(dataframe) % tf - 1

        dataframe[["open", "high", "low", "close", "volume"]] = dataframe[
            ["open", "high", "low", "close", "volume"]].astype(float)

        # Creating a new dataframe so that the size of the rows of the new dataframe will be the same as the new columns
        df = dataframe.iloc[shift::tf].copy()

        # Iterating through candle data, and abstracting based on the highest, lowest and sum respectively.
        df['high'] = [max(dataframe['high'][i:tf + i]) for i in range(shift, len(dataframe['high']), tf)]
        df['low'] = [min(dataframe['low'][i:tf + i]) for i in range(shift, len(dataframe['low']), tf)]
        df['volume'] = [sum(dataframe['volume'][i:tf + i]) for i in range(shift, len(dataframe['volume']), tf)]

        df['timeframe'] = [tf]*len(df['volume'])

        # Selecting every nth value in the list, where n is the timeframe
        df['close'] = [dataframe['close'].iloc[i:tf + i].iloc[-1] for i in range(shift, len(dataframe['close']), tf)]

        # Dropping the last value, this gets rid of the candle that isn't complete until the end of the tf
        if drop_last_row:
            df.drop(df.tail(1).index, inplace=True)

        return df.reset_index().set_index(['symbol', 'timeframe', 'timestamp'])

    def get_all_binance_data(self, symbol, start_date, end_date=None, tf='1m'):
        list_symbol = self.client.get_historical_klines(symbol=symbol, interval=tf, start_str=start_date)
        df_symbol = pd.DataFrame(list_symbol)
        df_symbol.columns = ["timestamp", "open", "high", "low", "close", "volume", "timestamp_end", "", "", "", "", ""]

        ##Fixing Columns
        df_symbol['timestamp'] = [int(timestamp / 1000) for timestamp in df_symbol['timestamp']]
        df_symbol['date'] = [datetime.fromtimestamp(timestamp) for timestamp in df_symbol['timestamp'].values]
        df_symbol['timeframe'] = [tf[:-1]] * len(df_symbol)
        df_symbol['symbol'] = [symbol] * len(df_symbol)

        df_symbol[["open", "high", "low", "close", "volume"]] = df_symbol[
            ["open", "high", "low", "close", "volume"]].astype(
            float)
        df_symbol = df_symbol[['symbol', 'timeframe', 'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume']]
        df_symbol = df_symbol.set_index(["symbol", "timeframe", "timestamp"])

        start_date = str(df_symbol.iloc[0, 0])[:10]

        string = f"Binance {symbol} {tf} data from {start_date} to {str(datetime.now())[:10]}.csv"
        print(string)
        # df_symbol.to_csv(string)

        return df_symbol