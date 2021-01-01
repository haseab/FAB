import time
import pandas as pd
from datetime import datetime
import numpy as np


class DataLoader():
    """
    Responsible for all ETL related tasks. Loads data from csv, fetches data from Binance API.

    Attributes
    -----------
    None

    Methods
    ------------
    load_csv
    get_range
    timeframe_setter

    Please look at each method for descriptions
    """

    def load_csv(self, csvUrl: str) -> pd.DataFrame:
        """Function used to load 1-minute historical candlestick data with a given csv url
            The important columns are the ones that create the candlestick (open, high, low, close) """
        # Reading CSV File containing 1 min candlestick data
        data = pd.read_csv(csvUrl, index_col='Timestamp')
        # Converting Timestamp numbers into a new column of readable dates
        data['Datetime'] = [datetime.fromtimestamp(i) for i in data.index]
        data[["Open", "High", "Low", "Close", "Volume"]] = data[["Open", "High", "Low", "Close", "Volume"]].astype(
            float)
        data = data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        return data

    def get_range(self, dataframe: pd.DataFrame, start_date: str = '2018-4-1',
                  end_date: str = '2018-5-1') -> pd.DataFrame:
        """Returns the range of 1-min data within specified start & end date from the entire dataset

            Parameters
            ----------
            dataframe: pd.DataFrame object with a Timestamp as its inde
            start_date: date in the format of YYYY-MM-DD format
            end_date: date in the format of YYYY-MM-DD format

            Returns: dataframe
        """
        start_date = int(time.mktime(datetime.strptime(start_date, "%Y-%m-%d").timetuple()))
        end_date = int(time.mktime(datetime.strptime(end_date, "%Y-%m-%d").timetuple()))
        return dataframe.loc[start_date:end_date]

    def timeframe_setter(self, dataframe: pd.DataFrame, tf: int, shift: int = None) -> pd.DataFrame:
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
                    >>> dataframe['Close'] = [4.50, 4.60, 4.65, 4.44, 4.21, 4.54, 4.10]
                    >>> timeFrame = 2
                    >>> timeframe_setter(df, 2)
                    >>> df['Close']
                    [4.50, 4.65, 4.21, 4.10]

                    >>> df['Close'] = [4.50, 4.60, 4.65, 4.44, 4.21, 4.54, 4.10]
                    >>> timeFrame = 3
                    >>> timeframe_setter(df, 3)
                    >>> df['Close']
                    [4.50, 4.44, 4.10]

            Open - Same rules as Close

            High - Get the maximum 1-min high value given the range of the timeframe
                 Ex.
                    >>> dataframe['High'] = [4.50, 4.60, 4.65, 4.44, 4.21, 4.54, 4.10]
                    >>> timeFrame = 2
                    >>> timeframe_setter(df, 2)
                    >>> df['Close']
                    [4.60, 4.65, 4.44, 4.54]

                    >>> df['High'] = [4.50, 4.60, 4.65, 4.44, 4.21, 4.54]
                    >>> timeFrame = 3
                    >>> timeframe_setter(df, 3)
                    >>> df['High']
                    [4.65, 4.54]

            Low - Same rules as 'High', but instead the minimum of that range

            Volume - Same rules as "High", but instead the sum of that range

        If the range of tf is not even (such as having a tf=2 but only 5 elements), then the
        last value will be dropped

        Returns: dataframe
        """
        if shift == None:
            shift = tf - len(dataframe) % tf - 1

        dataframe[["Open", "High", "Low", "Close", "Volume"]] = dataframe[
            ["Open", "High", "Low", "Close", "Volume"]].astype(float)

        # Creating a new dataframe so that the size of the rows of the new dataframe will be the same as the new columns
        df = dataframe.iloc[shift::tf].copy()

        # Iterating through a range of "tf" minute candle data, and abstracting based on the higest, lowest and sum respectively.
        df['High'] = [max(dataframe['High'][i:tf + i]) for i in range(shift, len(dataframe['High']), tf)]
        df['Low'] = [min(dataframe['Low'][i:tf + i]) for i in range(shift, len(dataframe['Low']), tf)]
        df['Volume'] = [sum(dataframe['Volume'][i:tf + i]) for i in range(shift, len(dataframe['Volume']), tf)]

        # Selecting every nth value in the list, where n is the timeframe
        df['Close'] = [dataframe['Close'].iloc[i:tf + i].iloc[-1] for i in range(shift, len(dataframe['Close']), tf)]

        # Dropping the last value, this get's rid of the candle that isnt complete until the end of the tf
        df.drop(df.tail(1).index, inplace=True)

        return df

    def timeframe_setter_v2(self, dfraw: pd.DataFrame, tf: int, shift: int = None) -> pd.DataFrame:
        """
        Horizontal way of abstracting the data - WORK IN PROGRESS

        This way of abstracting data actually takes longer and more time, however it allows for
        complex cases in which not all data needs to have the same timeframe.

        """

        if shift == None:
            shift = tf - len(dfraw) % tf - 1

        start = time.time()
        shift, tf = 8, 77
        count = 0
        low, high = shift, shift + tf
        df2 = dfraw.copy().head(0)
        hidf = dfraw.loc[:, "High"]
        lodf = dfraw.loc[:, "Low"]
        while count < 1000:
            df2 = df2.append({"Datetime": dfraw.iloc[0, 0], "Open": dfraw.iloc[0, 1], "High": max(hidf[low:high]),
                              "Low": min(lodf[low:high]), "Close": dfraw.iloc[-1, 4]}, ignore_index=True)
            low += 77
            high += 77
            count += 1
        end = time.time()
        return df2