import math
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np


class Helper:
    """class that is filled with a bunch of helper methods needed to run the bot"""
    SECONDS_IN_A_MINUTE = 60

    @staticmethod
    def sig_fig(x: float, sig: int = 2) -> float:
        """ Rounds to the number of significant digits indicated"""
        return round(x, sig - math.ceil(math.log10(abs(x))))

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
    def into_dataframe(lst: list) -> pd.DataFrame:
        """Converts Binance response list into dataframe"""
        return pd.DataFrame(lst, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume",
                                          "Timestamp_end", "", "", "", "", ""]).set_index("Timestamp")

    @staticmethod
    def millisecond_timestamp_to_datetime(timestamp_list):
        return [datetime.fromtimestamp(second_timestamp / 1000) for second_timestamp in timestamp_list]

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
        ranges = np.ceil(np.linspace(0, tf * 235, num=split_number))

        # Converting all indices into integers and reversing the list
        ranges = [int(index) for index in ranges[::-1]]
        return ranges
