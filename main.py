def get_range(dataframe, start_date='2018-4-1', end_date='2018-5-1'):
    """Returns the range of min data that is required from the entire dataset"""

    datetimes = pd.date_range(start_date, end_date, freq='min').to_pydatetime()
    date_list = [date.strftime('%Y-%m-%d %H:%M:%S') for date in datetimes]
    common = dataframe[dataframe['Datetime'].isin(date_list)]
    return common


def timeframe_setter(dataframe, timeFrame=77):
    """Converts minute candlestick data into the timeframe(tf) of choice"""
    df = dataframe.iloc[::timeFrame].copy()
    df['High'] = np.array(
        [max(dataframe['High'][i:timeFrame + i]) for i in range(0, len(dataframe['High']), timeFrame)])
    df['Low'] = np.array([min(dataframe['Low'][i:timeFrame + i]) for i in range(0, len(dataframe['Low']), timeFrame)])
    df['Close'] = np.array([dataframe['Close'].iloc[timeFrame - 1 + i] for i in
                            range(0, len(dataframe['Close']) - timeFrame + 1, timeFrame)])
    return df
    #returns foo
    #ignore above, return bar