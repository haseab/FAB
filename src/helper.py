import math
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np


class Helper:
    """class that is filled with a bunch of helper methods needed to run the bot"""

    @staticmethod
    def sig_fig(x: float, sig: int = 2) -> float:
        """ Rounds to the number of significant digits indicated"""
        return round(x, sig - math.ceil(math.log10(abs(x))))

    @staticmethod
    def string_to_timestamp(date: str) -> int:
        """Converts String of form DD-MM-YY into millisecond timestamp"""
        return int(time.mktime(datetime.strptime(date, "%d/%m/%Y").timetuple())) * 1000

    @staticmethod
    def into_dataframe(lst: list) -> pd.DataFrame:
        """Converts Binance response list into dataframe"""
        return pd.DataFrame(lst, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume",
                                          "Timestamp_end", "", "", "", "", ""]).set_index("Timestamp")

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
        ranges = [int(i) for i in ranges[::-1]]
        return ranges
