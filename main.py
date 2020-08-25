def get_range(dataframe, start_date='2018-4-1', end_date='2018-5-1'):
    """Returns the range of min data that is required from the entire dataset"""

    datetimes = pd.date_range(start_date, end_date, freq='min').to_pydatetime()
    date_list = [date.strftime('%Y-%m-%d %H:%M:%S') for date in datetimes]
    common = dataframe[dataframe['Datetime'].isin(date_list)]
    return common

def foo():
    return "hi"
