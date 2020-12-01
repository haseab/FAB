class DataLoader():

    def __init__(self):
        pass

    def load_csv(self, csvUrl):
        """Function used to load 1-minute historical candlestick data with a given csv url
            The important columns are the ones that create the candlestick (open, high, low, close) """
        # Reading CSV File containing 1 min candlestick data
        data = pd.read_csv(csvUrl, index_col='Timestamp')
        # Converting Timestamp numbers into a new column of readable dates
        data['Datetime'] = np.array([datetime.fromtimestamp(i) for i in data.index])
        data = data[['Datetime', 'Open', 'High', 'Low', 'Close']]
        return data

    def get_range(self, dataframe, start_date='2018-4-1', end_date='2018-5-1'):
        """Returns the range of 1-min data within specified start & end date from the entire dataset
            The intention is to take create a range of datetime objects, convert the dates into the
            same format as the csv, and then use a pandas '.isin' method to get the range
        """
        # Creating a list of datetime objects starting form start date to the end date
        datetimes = pd.date_range(start_date, end_date, freq='min').to_pydatetime()
        # Converting datetime objects into a string, matching the CSV format
        date_list = [date.strftime('%Y-%m-%d %H:%M:%S') for date in datetimes]
        common = dataframe[dataframe['Datetime'].isin(date_list)]
        return common

    def timeframe_setter(self, dataframe, tf=77, shift=8):
        """ Vertical way of appending data
        Converts minute candlestick data into the timeframe(tf) of choice.
            Parameters:
                dataframe: the dataframe that is being passed as an argument

                tf: The combination of 1-min candles into one value. Number of 1-min candles combined
                        is the timeframe value itself.
                        The raw data is in a 1-min timeframe. Dataframe contains the following
                        columns: ['Open', 'High', 'Low, 'Close']. Converting to a 60 minute timeframe is
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

            If the range of tf is not even (such as having a tf=2 but only 5 elements), then the
            last value will be dropped

            Returns: dataframe
            """

        # Creating a new dataframe so that the size of the rows of the new dataframe will be the same as the new columns
        df = dataframe.iloc[shift::tf].copy()
        # Grouping the high and low values according to the range of the timeframe
        df['High'] = [max(dataframe['High'][i:tf + i]) for i in range(shift, len(dataframe['High']), tf)]
        df['Low'] = [min(dataframe['Low'][i:tf + i]) for i in range(shift, len(dataframe['Low']), tf)]
        # Dropping the last value
        df.drop(df.tail(1).index, inplace=True)
        # Selecting every nth value in the list, where n is the timeframe
        df['Close'] = np.array(
            [dataframe['Close'].iloc[tf - 1 + i] for i in range(shift, len(dataframe['Close']) - tf + 1, tf)])
        return df

    def timeframe_setter_v2(self, dataframe, tf=7, shift=8):
        """Horizontal way of appending the data """
        start = time.time()
        shift, tf = 8, 77
        count = 0
        low, high = shift, shift + tf
        df2 = dfraw.copy().head(0)
        hidf = dfraw.loc[:, "High"]
        lodf = dfraw.loc[:, "Low"]
        while count < 1000:
            df2 = df2.append({"Datetime": df.iloc[0, 0], "Open": df.iloc[0, 1], "High": max(hidf[low:high]),
                              "Low": min(lodf[low:high]), "Close": df.iloc[-1, 4]}, ignore_index=True)
            low += 77
            high += 77
            count += 1
        end = time.time()
        return df
