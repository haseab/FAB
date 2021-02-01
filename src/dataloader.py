import time
import pandas as pd
from datetime import datetime
from binance.client import Client
from helper import Helper


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

    def __init__(self):
        self.client = Client()

    def _load_csv(self, csv_url: str) -> pd.DataFrame:
        """Function used to load 1-minute historical candlestick data with a given csv url
            The important columns are the ones that create the candlestick (open, high, low, close) """
        # Reading CSV File containing 1 min candlestick data
        data = pd.read_csv(csv_url, index_col='Timestamp')
        # Converting Timestamp numbers into a new column of readable dates
        data['Datetime'] = [datetime.fromtimestamp(i) for i in data.index]
        data[["Open", "High", "Low", "Close", "Volume"]] = data[["Open", "High", "Low", "Close", "Volume"]].astype(
            float)
        data = data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
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

        seconds_in_a_minute = 60
        # Multiplies second timestamp to turn into millisecond timestamp (which binance uses)
        timestamp_adjust = 1000

        # Defining params to put in exchange API call
        startTime = (int(now) - seconds_in_a_minute * (
            start_minutes_ago) - 1) * timestamp_adjust  # Ex. 1609549634 -> in seconds
        endTime = int(now - seconds_in_a_minute * end_minutes_ago) * timestamp_adjust
        limit = abs(start_minutes_ago - end_minutes_ago)

        data = self.client.futures_klines(symbol=symbol, interval="1m", startTime=startTime, endTime=endTime,
                                          limit=limit)
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

        start_date = int(time.mktime(datetime.strptime(start_date, "%Y-%m-%d").timetuple()))
        end_date = int(time.mktime(datetime.strptime(end_date, "%Y-%m-%d").timetuple()))
        return dataframe.loc[start_date:end_date]

    def _timeframe_setter(self, dataframe: pd.DataFrame, tf: int, shift: int = None) -> pd.DataFrame:
        """ Vertical way of abstracting data
        Converts minute candlestick data into the timeframe(tf) of choice.
        Parameters
        -----------
        dataframe: the dataframe that is being passed as an argument

        tf: The combination of 1-min candles into one value. Number of 1-min candles combined
                is the timeframe value itself.
                The raw data is in a 1-min timeframe. Dataframe contains the following
                columns: ['Open', 'High', 'Low, 'Close']. Converting to a X minute timeframe is
                handled differently for every column of the candlestick:

            Close - Since all that matters is the close value every 'tf' minutes, you can skip
                every 'tf' minutes.
                Ex.
                    df['Close'] = pd.Series([4.50, 4.60, 4.65, 4.44, 4.21, 4.54, 4.10])
                    _timeframe_setter(df['Close'], 2) -> [4.50, 4.65, 4.21, 4.10]
                    _timeframe_setter(df['Close'], 3) -> [[4.50, 4.44, 4.10]

            Open - Same rules as Close

            High - Get the maximum 1-min high value given the range of the timeframe
                 Ex.
                     df['Close'] = pd.Series([4.50, 4.60, 4.65, 4.44, 4.21, 4.54, 4.10])
                    _timeframe_setter(df['High'], 2) ->  [4.60, 4.65, 4.44, 4.54]
                    _timeframe_setter(df['High'], 3) ->  [4.65, 4.54]

            Low - Same rules as 'High', but instead the minimum of that range

            Volume - Same rules as "High", but instead the sum of that range

        If the range of tf is not even (such as having a tf=2 but only 5 elements), then the
        last value will be dropped

        :return dataframe
        """
        if shift == None:
            # This is making sure that there it shifts so that the last tf candle includes the last 1-minute candle
            shift = tf - len(dataframe) % tf - 1

        dataframe[["Open", "High", "Low", "Close", "Volume"]] = dataframe[
            ["Open", "High", "Low", "Close", "Volume"]].astype(float)

        # Creating a new dataframe so that the size of the rows of the new dataframe will be the same as the new columns
        df = dataframe.iloc[shift::tf].copy()

        # Iterating through candle data, and abstracting based on the highest, lowest and sum respectively.
        df['High'] = [max(dataframe['High'][i:tf + i]) for i in range(shift, len(dataframe['High']), tf)]
        df['Low'] = [min(dataframe['Low'][i:tf + i]) for i in range(shift, len(dataframe['Low']), tf)]
        df['Volume'] = [sum(dataframe['Volume'][i:tf + i]) for i in range(shift, len(dataframe['Volume']), tf)]

        # Selecting every nth value in the list, where n is the timeframe
        df['Close'] = [dataframe['Close'].iloc[i:tf + i].iloc[-1] for i in range(shift, len(dataframe['Close']), tf)]

        # Dropping the last value, this gets rid of the candle that isn't complete until the end of the tf
        df.drop(df.tail(1).index, inplace=True)

        return df

    def _timeframe_setter_v2(self, df_raw: pd.DataFrame, tf: int, shift: int = None) -> pd.DataFrame:
        """
        WORK IN PROGRESS - Horizontal way of abstracting the data

        This way of abstracting data actually takes longer and more time, however it allows for
        complex cases in which not all data needs to have the same timeframe.

        """

        if shift == None:
            # This is making sure that there it shifts so that the last tf candle includes the last 1-minute candle
            shift = tf - len(df_raw) % tf - 1

        tf = 77
        count = 0
        low, high = shift, shift + tf
        df2 = df_raw.copy().head(0)
        hi_df = df_raw.loc[:, "High"]
        lo_df = df_raw.loc[:, "Low"]
        while count < 1000:
            df2 = df2.append({"Datetime": df_raw.iloc[0, 0], "Open": df_raw.iloc[0, 1], "High": max(hi_df[low:high]),
                              "Low": min(lo_df[low:high]), "Close": df_raw.iloc[-1, 4]}, ignore_index=True)
            low += 77
            high += 77
            count += 1

        return df2
