from trading_history import TradeHistory
from exceptions import Exceptions
import pandas as pd
from decimal import Decimal
from helper import Helper
import statistics
import time

class Analyzer:
    """
    Responsible for analysis of trading history.

    Attributes
    -----------
    trades:               [float]       - pnl scores of each indvidiual trade (Ex. [1.25,0.97,1.10,0.91]
    trade_history:        TradeHistory  - Object that contains list of Trade Objects
    pnl:                  float         - Cumulative pnl score (Ex. 1.59)
    gross_profit:         float         - Cumulative profit score (Ex.7.56)
    gross_loss:           float         - Cumulative loss score (Ex.0.25)
    largest_profit:       float
    largest_loss:         float
    longest_run:          float         - Cumulative pnl for consecutive profits (Ex. 1.64)
    longest_drawdown:     float         - Cumulative pnl for consecutive losses (Ex. 0.85)
    average_win:          float
    average_loss:         float
    trades_won:           int
    trades_lost:          int
    win_streak:           int
    lose_streak:          int
    self.initial_capital: float          - Balance before trading using strategy
    self.capital:         float          - Balance after trading using strategy


    Methods
    ------------
    calculate_longest_run
    calculate_longest_drawdown
    calculate_statistics
    summarize_statistics

    Please look at each method for descriptions
    """
    def __init__(self):
        self.trade_history = None
        self.candle_index = None
        self.trade_index = None
        self.peak_index = None
        self.candle_volatility_index = None
        self.candle_volume_index = None
        self.candle_pps_index = None

    def calculate_longest_run(self, trade_index: pd.DataFrame) -> (float, int):
        """
        Takes a list of floats (e.g. [1.05, 0.98, 1.11, 1.01, 0.78] and
        calculates the longest consecutive chain of numbers greater than 1.
        Each float represents 1 + the profit margin of a trade.

        Ex. [1.05, 0.98, 1.11, 1.01, 0.78] -> (1.1211, 2)
        Ex. [1.06, 1.05, 1.10, 1.05, 0.88] -> (1.2855, 4)

        :return tuple of the product of all valid consecutive numbers and the streak number
        """
        trade_index = trade_index['profitability']
        longest_run, longest_counter = 1, 0
        current_run, current_counter = 1, 0

        for i in range(len(trade_index)):
            if trade_index.iloc[i] < 1:
                longest_run = max(longest_run, current_run)
                longest_counter = max(longest_counter, current_counter)
                current_run, current_counter = 1, 0
                continue
            current_run *= trade_index.iloc[i]
            current_counter += 1
        longest_run = max(longest_run, current_run)
        longest_counter = max(longest_counter, current_counter)
        return longest_run, longest_counter

    def calculate_longest_drawdown(self, trade_index: [float]) -> (float, int):
        """
        Takes a list of inputted trade results (e.g. [1.05, 0.98, 1.11, 1.01, 0.78] and
        calculates the longest consecutive chain of numbers less than 1.
        Each float represents 1 + the profit margin of a trade.

        Ex. [1.05, 0.98, 1.11, 1.01, 0.78] -> (0.78, 1)
        Ex. [1.06, 1.05, 1.10, 0.95, 0.88] -> (0.8624, 2)

        :return tuple of the product of all valid consecutive numbers and the streak number
        """
        trade_index = trade_index['profitability']
        longest_drawdown, longest_counter = 1, 0
        current_drawdown, current_counter = 1, 0

        for i in range(len(trade_index)):
            if trade_index.iloc[i] > 1:
                longest_drawdown = min(longest_drawdown, current_drawdown)
                longest_counter = max(longest_counter, current_counter)
                current_drawdown, current_counter = 1, 0
                continue
            current_drawdown *= trade_index.iloc[i]
            current_counter += 1
        longest_drawdown = min(longest_drawdown, current_drawdown)
        longest_counter = max(longest_counter, current_counter)
        return longest_drawdown, longest_counter

    def generate_general_indices(self, detailed_th):
        self.candle_volatility_index = pd.DataFrame(detailed_th['high']/detailed_th['low'], columns=['volatility'])
        self.candle_volume_index = pd.DataFrame(detailed_th['volume'])
        self.candle_pps_index = pd.DataFrame(detailed_th['close'] / detailed_th['volume'], columns=['pps'])
        return "Calculated all general indices"

    def generate_long_indices(self, detailed_th, commission = 0.9996):
        detailed_th = detailed_th[detailed_th['side'] == "Long"]
        candle_long_index = pd.DataFrame(detailed_th['close']/detailed_th['open'], columns=['pnl'])
        candle_long_index['side'] = ['Long' for i in range(len(candle_long_index))]
        tf = detailed_th['tf'].iloc[0]

        peak_long_index, trade_long_index = pd.DataFrame(), pd.DataFrame()
        for tid in detailed_th.index.unique('tid'):
            subset_df = detailed_th.loc[tid, :]
            first_index = subset_df.index[0]
            last_index = subset_df.index[-1]

            if subset_df.loc[first_index, 'side'] != "Long":
                continue

            # Getting Trade Long indices
            side = subset_df.loc[first_index, 'side']
            enter_trade = subset_df.loc[first_index, 'enter_price']
            exit_trade = subset_df.loc[last_index, 'exit_price']
            profitability = Helper.calculate_long_profitability(enter_trade, exit_trade, commission)
            trade_long_index = trade_long_index.append(pd.DataFrame([[tid, side, first_index, enter_trade, last_index,
                    exit_trade, round(profitability, 5)]], columns=['tid', 'side', "enter_id", "enter_trade",
                                                                    "exit_id", "exit_trade", "profitability"]))
            # Getting Peak Long Indices
            if len(subset_df) <= tf:
                peak_id = subset_df['high'].idxmax()
            else:
                peak_id = subset_df['high'][tf:].idxmax()
            peak_price = subset_df.loc[peak_id, 'high']
            trough_id = subset_df['low'].idxmin()
            trough_price = subset_df.loc[trough_id, 'low']
            peak_long_index = peak_long_index.append(pd.DataFrame([[tid, peak_id, peak_price, trough_id, trough_price]],
                                        columns=['tid', "peak_id", "peak_price", "trough_id", "trough_price"]))

        return candle_long_index, trade_long_index, peak_long_index

    def generate_short_indices(self, detailed_th, commission = 0.9996):
        detailed_th = detailed_th[detailed_th['side'] == "Short"]
        candle_short_index = pd.DataFrame(2 - detailed_th['close']/detailed_th['open'], columns=['pnl'])
        candle_short_index['side'] = ['Short' for i in range(len(candle_short_index))]
        tf = detailed_th['tf'].iloc[0]

        peak_short_index, trade_short_index = pd.DataFrame(), pd.DataFrame()

        for tid in detailed_th.index.unique('tid'):
            subset_df = detailed_th.loc[tid, :]
            first_index = subset_df.index[0]
            last_index = subset_df.index[-1]

            if subset_df.loc[first_index, 'side'] != "Short":
                continue

            # Getting Trade Short indices
            side = subset_df.loc[first_index, 'side']
            enter_trade = subset_df.loc[first_index, 'enter_price']
            exit_trade = subset_df.loc[last_index, 'exit_price']
            profitability = Helper.calculate_short_profitability(enter_trade, exit_trade, commission)
            trade_short_index = trade_short_index.append(pd.DataFrame([[tid, side, first_index, enter_trade, last_index,
                    exit_trade, round(profitability, 5)]], columns=['tid', "side", "enter_id", "enter_trade",
                                                                    "exit_id", "exit_trade", "profitability"]))
            # Getting Peak Short Indices
            if len(subset_df) <= tf:
                peak_id = subset_df['low'].idxmin()
            else:
                peak_id = subset_df['low'][tf:].idxmin()
            peak_price = subset_df.loc[peak_id, 'low']
            trough_id = subset_df['high'].idxmax()
            trough_price = subset_df.loc[trough_id, 'high']
            peak_short_index = peak_short_index.append(pd.DataFrame([[tid, peak_id, peak_price, trough_id, trough_price]],
                                        columns=['tid', "peak_id", "peak_price", "trough_id", "trough_price"]))

        return candle_short_index, trade_short_index, peak_short_index

    def generate_all_indices(self, detailed_th):
        """Detailed Trading History needs to have the indices [['tid', 'candle_id']]"""
        candle_short_index, candle_long_index, trade_long_index, peak_long_index, trade_short_index, peak_short_index  = \
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.generate_general_indices(detailed_th)
        if len(detailed_th[detailed_th['side'] == "Short"]) != 0:
            candle_short_index, trade_short_index, peak_short_index = self.generate_short_indices(detailed_th)
        if len(detailed_th[detailed_th['side'] == "Long"]) != 0:
            candle_long_index, trade_long_index, peak_long_index = self.generate_long_indices(detailed_th)
        self.candle_index = candle_long_index.append(candle_short_index).sort_values('tid')
        self.trade_index = trade_long_index.append(trade_short_index).set_index('tid').sort_values('tid')
        self.peak_index = peak_long_index.append(peak_short_index).set_index('tid').sort_values('tid')
        return self

    def get_trade_history(self):
        return self.trade_history

    def get_largest_profit(self, trade_index):
        return trade_index['profitability'].max()

    def get_largest_loss(self, trade_index):
        return trade_index['profitability'].min()

    def get_average_win(self, gross_profit, num_trades_won):
        return gross_profit ** (1 / num_trades_won) if num_trades_won != 0 else 1

    def get_average_loss(self, gross_loss, num_trades_lost):
        return gross_loss ** (1 / num_trades_lost) if num_trades_lost != 0 else 1

    def get_num_trades_won(self, trade_index):
        return len(trade_index[trade_index['profitability']>1])

    def get_num_trades_lost(self, trade_index):
        return len(trade_index[trade_index['profitability']<=1])

    def get_longest_run(self, trade_index):
        return self.calculate_longest_run(trade_index)[0]

    def get_longest_drawdown(self, trade_index):
        return self.calculate_longest_drawdown(trade_index)[0]

    def get_lose_streak(self, trade_index):
        return self.calculate_longest_drawdown(trade_index)[1]

    def get_win_streak(self, trade_index):
        return self.calculate_longest_run(trade_index)[1]

    def get_gross_profit(self, trade_index):
        trade_list = trade_index['profitability']
        profit_list = trade_list[trade_list > 1]
        gross_profit = profit_list.product()
        return gross_profit

    def get_gross_loss(self, trade_index):
        trade_list = trade_index['profitability']
        profit_list = trade_list[trade_list < 1]
        gross_loss = profit_list.product()
        return gross_loss

    def calculate_profitability(self, trade_index, median=True):
        if median:
            return trade_index['profitability'].median()
        return trade_index['profitability']

    def get_profit_rate(self, candle_index: pd.Series, mean=True):
        df_profit = candle_index[candle_index['pnl'] > 1]['pnl']
        if mean:
            mean_median = pd.Series()
            for tid in df_profit.index.unique('tid'):
                mean_median = mean_median.append(pd.Series(df_profit.loc[tid, :].median()))
            return mean_median.mean()
        return df_profit

    def get_loss_rate(self, candle_index: pd.Series, mean=True):
        df_profit = candle_index[candle_index['pnl'] < 1]['pnl']
        if mean:
            mean_median = pd.Series()
            for tid in df_profit.index.unique('tid'):
                mean_median = mean_median.append(pd.Series(df_profit.loc[tid, :].median()))
            return mean_median.mean()
        return df_profit

    def get_peak_profit_rate(self, candle_index, peak_index, mean=True):
        means = pd.Series()
        candle_index = candle_index[candle_index['pnl'] > 1]
        for tid in candle_index.index.unique('tid'):
            subset_df = candle_index.loc[tid, :]
            subset_df_mean = subset_df[subset_df.index <= peak_index.loc[tid, 'peak_id']].median()
            means = means.append(subset_df_mean)
        if mean:
            return means.mean()
        return means

    def get_peak_loss_rate(self, candle_index, peak_index, mean=True):
        medians = pd.Series()
        candle_index = candle_index[candle_index['pnl'] < 1]
        for tid in candle_index.index.unique('tid'):
            subset_df = candle_index.loc[tid, :]
            subset_df_mean = subset_df[subset_df.index <= peak_index.loc[tid, 'peak_id']].median()
            medians = medians.append(subset_df_mean)
        if mean:
            return medians.mean()
        return medians

    def get_pps_rate(self, pps_index, median=True):
        """PPS - Price per share"""
        if median:
            return pps_index['pps'].median()
        return pps_index['pps']

    def get_peak_pps_rate(self, pps_index, peak_index, mean=True):
        """PPS - Price per share"""
        medians = pd.Series()
        for tid in pps_index.index.unique('tid'):
            subset_df = pps_index.loc[tid, :]
            subset_df_mean = subset_df[subset_df.index <= peak_index.loc[tid, 'peak_id']].median()
            medians = medians.append(subset_df_mean)
        if mean:
            return medians.mean()
        return medians

    def get_volatility_rate(self, volatility_index, median=True):
        if median:
            return volatility_index['volatility'].median()
        return volatility_index['volatility']

    def get_volume_rate(self, volume_index, median=True):
        if median:
            return volume_index['volume'].median()
        return volume_index['volume']

    def get_peak_volume_rate(self, volume_index, peak_index, mean=True):
        medians = pd.Series()
        for tid in volume_index.index.unique('tid'):
            subset_df = volume_index.loc[tid, :]
            subset_df_mean = subset_df[subset_df.index <= peak_index.loc[tid, 'peak_id']].median()
            medians = medians.append(subset_df_mean)
        if mean:
            return medians.mean()
        return medians

    def get_num_candles_to_peak(self, candle_index, peak_index, median=True):
        means = pd.DataFrame(columns=['tid', 'num_candles'])
        for tid in candle_index.index.unique('tid'):
            subset_df = candle_index.loc[tid, :]
            subset_max = subset_df[subset_df.index <= peak_index.loc[tid, 'peak_id']]
            means = means.append(pd.DataFrame([[tid, len(subset_max)]], columns=['tid', 'num_candles']))
        if median:
            return means['num_candles'].median()
        return means.set_index('tid')

    def get_peak_unrealized_profit(self, trade_index, peak_index, commission, median=True):
        means = pd.DataFrame(columns=["tid", 'pnl', 'peak_profit'])
        unrealized = None
        for tid in trade_index.index.unique('tid'):
            if trade_index.loc[tid, 'side'] == "Long":
                unrealized = (commission**2) * (peak_index.loc[tid, 'peak_price']/ trade_index.loc[tid, 'enter_trade'])
            if trade_index.loc[tid, 'side'] == "Short":
                unrealized = (commission**2) * (2-(peak_index.loc[tid, 'peak_price'] / trade_index.loc[tid, 'enter_trade']))
            means = means.append(pd.DataFrame([[tid, trade_index.loc[tid, 'profitability'], unrealized]], columns=["tid", 'pnl', 'peak_profit']))
        if median:
            return means['peak_profit'].median()
        return means.reset_index(drop=True)

    def get_peak_unrealized_loss(self, trade_index, peak_index, commission, median=True):
        means = pd.DataFrame(columns=["tid", 'pnl', 'peak_loss'])
        unrealized = None
        for tid in trade_index.index.unique('tid'):
            if trade_index.loc[tid, 'side'] == "Long":
                unrealized = (commission**2) * peak_index.loc[tid, 'trough_price'] / trade_index.loc[tid, 'enter_trade']
            if trade_index.loc[tid, 'side'] == "Short":
                unrealized = (commission**2) * (2-(peak_index.loc[tid, 'trough_price'] / trade_index.loc[tid, 'enter_trade']))
            means = means.append(pd.DataFrame([[tid, trade_index.loc[tid, 'profitability'], unrealized]], columns=["tid", 'pnl', 'peak_loss']))
        if median:
            return means['peak_loss'].median()
        return means.reset_index(drop=True)

    def get_volume(self, volume_index, median=True):
        means = pd.Series()
        for tid in volume_index.index.unique('tid'):
            volume = volume_index.loc[tid, :].sum()
            means = means.append(pd.Series(volume))
        if median:
            return means.median()
        return means

    def get_peak_volume(self, volume_index, peak_index, median=True):
        means = pd.Series()
        for tid in volume_index.index.unique('tid'):
            subset_df = volume_index.loc[tid, :]
            subset_max = subset_df[subset_df.index <= peak_index.loc[tid, 'peak_id']].sum()
            means = means.append(pd.Series(subset_max))
        if median:
            return means.median()
        return means

    def get_captured_profit(self, average_win, peak_unrealized_profit_index):
        average_peak_profit = peak_unrealized_profit_index.product()**1/len(peak_unrealized_profit_index)
        return average_win/average_peak_profit

    def get_captured_loss(self, average_loss, peak_unrealized_loss_index):
        average_peak_loss = peak_unrealized_loss_index.product() ** 1 / len(peak_unrealized_loss_index)
        return average_loss/average_peak_loss

    def green_red_candle_ratio(self, candle_index):
        num_green_candles = len(candle_index[candle_index['pnl'] > 1])
        num_red_candles = len(candle_index[candle_index['pnl'] < 1])
        return num_green_candles/num_red_candles

    def get_minimum_rrr(self, longest_run, longest_drawdown):
        return round((longest_run - 1) / (1 - longest_drawdown), 5)

    def get_average_rrr(self, average_win, average_loss):
        if average_win == 1 or average_loss == 1:
            return -0.001
        return (average_win-1)/(1-average_loss)

    def get_unrealized_rrr(self, peak_unrealized_profit_index, peak_unrealized_loss_index):
        df = (peak_unrealized_profit_index-1)/(1-peak_unrealized_loss_index)
        return df.abs().mean()

    def get_amount_of_data(self, trade_index):
        return len(trade_index)

    def get_trade_activity_ratio(self, candle_index, df):
        return len(candle_index)/len(df)

#
# longest_run = b.analyzer.get_longest_drawdown(trade_index)
# longest_drawdown = b.analyzer.get_longest_run(trade_index)
# min_rrr = b21.analyzer.get_minimum_rrr(longest_run, longest_drawdown)