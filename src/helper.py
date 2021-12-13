import math
import time
from datetime import datetime, timedelta
from os import stat

import dateparser
import numpy as np
import pandas as pd
from ib_insync import util


class Helper:
    """class that is filled with a bunch of helper methods needed to run the bot"""
    SECONDS_IN_A_MINUTE = 60

    @staticmethod
    def sig_fig(x: float, sig: int = 2) -> float:
        """ Rounds to the number of significant digits indicated"""
        return round(x, sig - math.ceil(math.log10(abs(x))))
    @staticmethod
    def calculate_short_profitability(enter_price, exit_price, commission):
        return (commission ** 2) * (2 - exit_price / enter_price)

    @staticmethod
    def calculate_long_profitability(enter_price, exit_price, commission):
        return (commission ** 2) * (exit_price / enter_price)

    @staticmethod
    def sleep_for(seconds, partition=20):
        print(f"Sleeping {seconds} seconds...")
        for i in range(partition):
            time.sleep(seconds / partition)

    @staticmethod
    def current_minute_datetime():
        now = datetime.now()
        now = datetime(year=now.year, month=now.month, day=now.day, hour=now.hour, minute=now.minute)
        return now

    @staticmethod
    def factor_to_percentage(list_of_factors):
        list_of_percentages = []
        for factor in list_of_factors:
            if factor < 1:
                list_of_percentages.append(-round((1 - factor) * 100, 4))
            else:
                list_of_percentages.append(round((factor - 1) * 100, 4))

        return list_of_percentages

    @staticmethod
    def slope(series, i):
        return (series[i]-series[i-3]) / 3

    @staticmethod
    def max_index(index_list):
        max_index = 0
        maximum = index_list[max_index]

        for i in range(len(index_list)):
            if index_list[i] > maximum:
                max_index = i
                maximum = index_list[i]
        return max_index, maximum

    @staticmethod
    def timestamp_object_to_string(date_list):
        return np.array([str(date_list.iloc[i]) for i in range(len(date_list))])

    @staticmethod
    def string_to_timestamp(date: str, adjust=1) -> int:
        """Converts String of form DD-MM-YY into millisecond timestamp"""
        return int(time.mktime(datetime.strptime(date, "%Y-%m-%d").timetuple())) * adjust

    def minutes_ago_to_timestamp(self, minutes_ago, from_timestamp, adjust=1):
        # Multiplies second timestamp to turn into millisecond timestamp (which binance uses)
        return int(from_timestamp - self.SECONDS_IN_A_MINUTE * minutes_ago - 1) * adjust

    @staticmethod
    def change_in_clock_minute():
        return round(time.time() % 60, 1) == 0

    @staticmethod
    def find_greatest_divisible_timeframe(tf):
        divisible_list = [1, 3, 5, 15, 30, 60, 120, 240, 360, 480]
        for divisible_tf in reversed(divisible_list):
            if tf % divisible_tf == 0:
                return divisible_tf
        return False


    @staticmethod
    def into_dataframe(data: list, symbol, tf, qtrade=False) -> pd.DataFrame:
        """Converts Binance response list into dataframe"""
        crypto = 'USDT' in symbol or 'BUSD' in symbol

        if crypto:
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume",
                                            "", "", "", "", "", ""])
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(
                float)
            df['date'] = Helper.millisecond_timestamp_to_datetime(df['timestamp'])

        else:
            if qtrade:
                # Questrade Data Handling 
                df = pd.DataFrame(data).drop('start', axis=1).rename(columns = {'end': 'date'})
                df['date'] = [pd.Timestamp(date).tz_localize(None) for date in df['date']]
                df['timestamp'] = Helper.datetime_to_millisecond_timestamp(df['date'])
            else:
                # IBKR Data Handling
                df = util.df(data).drop(['average', 'barCount'], axis=1)
                df['timestamp'] = Helper.datetime_to_millisecond_timestamp(df['date'])

        df['symbol'] = [symbol]*len(data)
        df['tf'] = [tf]*len(data)

        return df[['symbol', 'tf', 'timestamp', 'date', 'open', 'high','low', 'close', 'volume']].set_index("timestamp")

    @staticmethod
    def finviz_market_cap_str_to_float(df):
        new_list = []
        for market_cap in df['Market Cap']:
            if 'B' in market_cap:
                new_list.append(float(market_cap[:-1])*10**9)
            elif 'M' in market_cap:
                new_list.append(float(market_cap[:-1])*10**6)
            else:
                new_list.append(0)
        df['Market Cap'] = new_list
        return df

    from functools import wraps

    @staticmethod
    def timeit(method):
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            if 'log_time' in kw:
                name = kw.get('log_name', method.__name__.upper())
                kw['log_time'][name] = int((te - ts) * 1000)
            else:
                print('%r  %2.2f ms' % \
                    (method.__name__, (te - ts) * 1000))
            return result
        return timed
    
    @staticmethod
    def datetime_from_tf(tf, daily_1m_candles=525, max_candles_needed=235, qtrade = False):
        how_many_days_ago = max_candles_needed*tf//daily_1m_candles + 1
        now = datetime.now()
        parsed_start, parsed_end = now-timedelta(days=how_many_days_ago), now
        if qtrade:
            parsed_start, parsed_end = parsed_start.strftime('%Y-%m-%d %H:%M:%S.%f'), parsed_end.strftime('%Y-%m-%d %H:%M:%S.%f')
        return parsed_start, parsed_end

    @staticmethod
    def millisecond_timestamp_to_datetime(timestamp_list):
        return [datetime.fromtimestamp(second_timestamp / 1000) for second_timestamp in timestamp_list]

    def datetime_to_millisecond_timestamp(datetime_list):
        return (pd.to_datetime(datetime_list) - pd.Timestamp("1970-01-01").tz_localize(None)) // pd.Timedelta('1s')

    @staticmethod
    def calculate_minute_disparity(df: pd.DataFrame, tf: int) -> float:
        """Calculates the difference (in minutes) of old the current dataframe is with respect to the live data"""
        # Getting the last date on your current dataset
        last_minute = df.iloc[-1]["Datetime"].to_pydatetime()
        current_minute = datetime.now()
        #     print(current_minute,last_minute, timedelta(minutes=tf))
        # Performing calculation to get the difference
        diff = (current_minute - last_minute - timedelta(minutes=tf)).seconds / 60
        return diff

    @staticmethod
    def determine_candle_positions(max_candles_needed, tf):
        # Ex. The 231 MA needs 231 candles of data to work. We use 4 more candles for safety.
        max_candles_needed += 4

        # Formula for determining how many discrete 1000-candle sets are needed
        split_number = math.ceil(tf * max_candles_needed / 1000) + 1

        # Determining the exact indices of when the set boundaries end
        ranges = np.ceil(np.linspace(0, tf * max_candles_needed, num=split_number))

        # Converting all indices into integers and reversing the list
        ranges = [int(index) for index in ranges[::-1]]
        return ranges
