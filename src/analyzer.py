from trading_history import TradeHistory
from exceptions import Exceptions
import pandas as pd
from decimal import Decimal
from helper import Helper
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
        self.capital = None
        self.initial_capital = None
        self.trades = None
        self.trade_history = None

    def get_capital(self):
        return self.capital

    def get_initial_capital(self):
        return self.initial_capital

    def get_trades(self):
        return self.trades

    def get_trade_history(self):
        return self.trade_history

    def get_largest_profit(self, trades):
        return max(trades)

    def get_largest_loss(self, trades):
        return min(trades)

    def get_average_win(self, gross_profit, num_trades_won):
        return gross_profit ** (1 / num_trades_won) if num_trades_won != 0 else 1

    def get_average_loss(self, gross_loss, num_trades_lost):
        return gross_loss ** (1 / num_trades_lost) if num_trades_lost != 0 else 1

    def get_num_trades_won(self, trade_index):
        return len(trade_index[trade_index['profitability']>1])

    def get_num_trades_lost(self, trade_index):
        return len(trade_index[trade_index['profitability']<=1])

    def get_longest_run(self, trades):
        return self.calculate_longest_run(trades)[0]

    def get_longest_drawdown(self, trades):
        return self.calculate_longest_drawdown(trades)[0]

    def get_lose_streak(self, trades):
        return self.calculate_longest_drawdown(trades)[1]

    def get_win_streak(self, trades):
        return self.calculate_longest_run(trades)[1]

    def get_pnl(self, trades):
        sums = 1
        for trade in trades:
            sums *= trade
        return sums

    def get_gross_profit(self, trades):
        sums = 1
        for trade in trades:
            if trade > 1:
                sums *= trade
        return sums

    def get_gross_loss(self, trades):
        sums = 1
        for trade in trades:
            if trade < 1:
                sums *= trade
        return sums

    def calculate_profitability(self, trade_index):
        return trade_index['profitability']

    def calculate_short_profitability(self, enter_price, exit_price, commission):
        return (commission ** 2) * (2 - exit_price / enter_price)

    def calculate_long_profitability(self, enter_price, exit_price, commission):
        return (commission ** 2) * (exit_price / enter_price)

    def calculate_longest_run(self, trade_index: [float]) -> (float, int):
        """
        Takes a list of floats (e.g. [1.05, 0.98, 1.11, 1.01, 0.78] and
        calculates the longest consecutive chain of numbers greater than 1.
        Each float represents 1 + the profit margin of a trade.

        Ex. [1.05, 0.98, 1.11, 1.01, 0.78] -> (1.1211, 2)
        Ex. [1.06, 1.05, 1.10, 1.05, 0.88] -> (1.2855, 4)

        :return tuple of the product of all valid consecutive numbers and the streak number
        """
        longest_run, longest_counter = 1, 0
        current_run, current_counter = 1, 0

        for i in range(len(trade_index)):
            if trade_index[i] < 1:
                longest_run = max(longest_run, current_run)
                longest_counter = max(longest_counter, current_counter)
                current_run, current_counter = 1, 0
                continue
            current_run *= trade_index[i]
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
        longest_drawdown, longest_counter = 1, 0
        current_drawdown, current_counter = 1, 0

        for i in range(len(trade_index)):
            if trade_index[i] > 1:
                longest_drawdown = min(longest_drawdown, current_drawdown)
                longest_counter = max(longest_counter, current_counter)
                current_drawdown, current_counter = 1, 0
                continue
            current_drawdown *= trade_index[i]
            current_counter += 1
        longest_drawdown = min(longest_drawdown, current_drawdown)
        longest_counter = max(longest_counter, current_counter)
        return longest_drawdown, longest_counter


    def get_general_indices(self,detailed_th):
        self.candle_volatility_index = pd.DataFrame(detailed_th['high']/detailed_th['low'], columns=['volatility'])
        self.candle_volume_index = pd.DataFrame(detailed_th['volume'])
        self.candle_pps_index = pd.DataFrame(detailed_th['close'] / detailed_th['volume'], columns=['pps'])
        return "Calculated all general indices"

    def get_long_indices(self, detailed_th, commission = 0.9996):
        detailed_th = detailed_th[detailed_th['side'] == "Long"]
        self.candle_long_index = pd.DataFrame(detailed_th['close']/detailed_th['open'], columns=['pnl'])
        peak_long_index, trade_long_index = pd.DataFrame(), pd.DataFrame()

        for tid in detailed_th.index.unique('trade_id'):
            subset_df = detailed_th.loc[tid, :]
            first_index = subset_df.index[0]
            last_index = subset_df.index[-1]

            if subset_df.loc[first_index, 'side'] != "Long":
                continue

            # Getting Peak Long Indices
            peak_id = subset_df['high'].idxmax()
            peak_price = subset_df.loc[peak_id, 'high']
            trough_id = subset_df['low'].idxmin()
            trough_price = subset_df.loc[trough_id, 'low']
            peak_long_index = peak_long_index.append(pd.DataFrame([[tid, peak_id, peak_price, trough_id, trough_price]],
                                        columns=['trade_id', "peak_id", "peak_price", "trough_id", "trough_price"]))

            # Getting Trade Long indices
            side = subset_df.loc[first_index, 'side']
            enter_trade = subset_df.loc[first_index, 'open']
            exit_trade = subset_df.loc[last_index, 'open']
            profitability = self.calculate_long_profitability(enter_trade, exit_trade, commission)
            trade_long_index = trade_long_index.append(pd.DataFrame([[tid, side, first_index, enter_trade, last_index,
                    exit_trade, round(profitability, 6)]], columns=['trade_id', 'side', "enter_id", "enter_trade",
                                                                    "exit_id", "exit_trade", "profitability"]))

            self.peak_long_index = peak_long_index.set_index('trade_id')
            self.trade_long_index = trade_long_index.set_index('trade_id')

        return "Calculated all long indices"

    def get_short_indices(self, detailed_th, commission = 0.9996):
        detailed_th = detailed_th[detailed_th['side'] == "Short"]
        self.candle_short_index = pd.DataFrame(2 - detailed_th['close']/detailed_th['open'], columns=['pnl'])
        peak_short_index, trade_short_index = pd.DataFrame(), pd.DataFrame()

        for tid in detailed_th.index.unique('trade_id'):
            subset_df = detailed_th.loc[tid, :]
            first_index = subset_df.index[0]
            last_index = subset_df.index[-1]

            if subset_df.loc[first_index, 'side'] != "Short":
                continue

            # Getting Peak Short Indices
            peak_id = subset_df['low'].idxmin()
            peak_price = subset_df.loc[peak_id, 'low']
            trough_id = subset_df['high'].idxmax()
            trough_price = subset_df.loc[trough_id, 'high']
            peak_short_index = peak_short_index.append(pd.DataFrame([[tid, peak_id, peak_price, trough_id, trough_price]],
                                        columns=['trade_id', "peak_id", "peak_price", "trough_id", "trough_price"]))

            # Getting Trade Short indices
            side = subset_df.loc[first_index, 'side']
            enter_trade = subset_df.loc[first_index, 'open']
            exit_trade = subset_df.loc[last_index, 'open']
            profitability = self.calculate_short_profitability(enter_trade, exit_trade, commission)
            trade_short_index = trade_short_index.append(pd.DataFrame([[tid, side, first_index, enter_trade, last_index,
                    exit_trade, round(profitability, 6)]], columns=['trade_id', "side", "enter_id", "enter_trade",
                                                                    "exit_id", "exit_trade", "profitability"]))

            self.peak_short_index = peak_short_index.set_index('trade_id')
            self.trade_short_index = trade_short_index.set_index('trade_id')

        return "Calculated all long indices"

    def get_all_indices(self, detailed_th):
        self.get_general_indices(detailed_th)
        self.get_short_indices(detailed_th)
        self.get_long_indices(detailed_th)
        return self

    def get_average_profit_rate(self, pnl_index: pd.Series):
        df_profit = pnl_index[pnl_index['pnl'] > 1]['pnl']
        return df_profit

    def get_average_loss_rate(self, pnl_index: pd.Series):
        df_profit = pnl_index[pnl_index['pnl'] < 1]['pnl']
        return df_profit

    def get_average_peak_profit_rate(self, pnl_index, peak_index):
        means = pd.Series()
        pnl_index = pnl_index[pnl_index['pnl'] > 1]
        for tid in pnl_index.index.unique('trade_id'):
            subset_df = pnl_index.loc[tid, :]
            subset_df_mean = subset_df[subset_df.index <= peak_index.loc[tid, 'peak_id']].mean()
            means = means.append(subset_df_mean)
        return means

    def get_average_peak_loss_rate(self, pnl_index, peak_index):
        means = pd.Series()
        pnl_index = pnl_index[pnl_index['pnl'] < 1]
        for tid in pnl_index.index.unique('trade_id'):
            subset_df = pnl_index.loc[tid, :]
            subset_df_mean = subset_df[subset_df.index <= peak_index.loc[tid, 'peak_id']].mean()
            means = means.append(subset_df_mean)
        return means

    def get_average_pps_rate(self, pps_index, peak_index):
        """PPS - Price per share"""
        return pps_index['pps']

    def get_average_peak_pps_rate(self, pps_index, peak_index):
        """PPS - Price per share"""
        means = pd.Series()
        for tid in pps_index.index.unique('trade_id'):
            subset_df = pps_index.loc[tid, :]
            subset_df_mean = subset_df[subset_df.index <= peak_index.loc[tid, 'peak_id']].mean()
            means = means.append(subset_df_mean)
        return means

    def get_average_volatility_rate(self, volatility_index):
        return volatility_index['volatility']

    def get_average_volume_rate(self, volume_index):
        return volume_index['volume']

    def get_average_peak_volume_rate(self, volume_index, peak_index):
        means = pd.Series()
        for tid in volume_index.index.unique('trade_id'):
            subset_df = volume_index.loc[tid, :]
            subset_df_mean = subset_df[subset_df.index <= peak_index.loc[tid, 'peak_id']].mean()
            means = means.append(subset_df_mean)
        return means

    def get_average_num_candles_to_peak(self, pnl_index, peak_index):
        means = pd.Series()
        for tid in pnl_index.index.unique('trade_id'):
            subset_df = pnl_index.loc[tid, :]
            subset_max = subset_df[subset_df.index <= peak_index.loc[tid, 'peak_id']]
            means = means.append(pd.Series(len(subset_max)))
        return means

    def get_average_peak_unrealized_profit(self, trade_index, peak_index):
        means = pd.DataFrame()
        unrealized = None
        for tid in trade_index.index.unique('trade_id'):
            if trade_index.loc[tid, 'side'] == "Long":
                unrealized = peak_index.loc[tid, 'peak_price'] / trade_index.loc[tid, 'enter_trade']
            if trade_index.loc[tid, 'side'] == "Short":
                unrealized = 2-(peak_index.loc[tid, 'peak_price'] / trade_index.loc[tid, 'enter_trade'])
            means = means.append(pd.DataFrame([[tid, trade_index.loc[tid, 'profitability'], unrealized]], columns = ["tid", 'profit', 'peak_profit']))
        return means

    def get_average_peak_unrealized_loss(self, trade_index, peak_index):
        means = pd.Series()
        unrealized = None
        for tid in trade_index.index.unique('trade_id'):
            if trade_index.loc[tid, 'side'] == "Long":
                unrealized = peak_index.loc[tid, 'trough_price'] / trade_index.loc[tid, 'enter_trade']
            if trade_index.loc[tid, 'side'] == "Short":
                unrealized = 2-(peak_index.loc[tid, 'trough_price'] / trade_index.loc[tid, 'enter_trade'])
            means = means.append(pd.Series(unrealized))
        return means

    def get_average_volume(self, volume_index):
        means = pd.Series()
        for tid in volume_index.index.unique('trade_id'):
            volume = volume_index.loc[tid, :].sum()
            means = means.append(pd.Series(volume))
        return means

    def get_average_peak_volume(self, volume_index, peak_index):
        means = pd.Series()
        for tid in volume_index.index.unique('trade_id'):
            subset_df = volume_index.loc[tid, :]
            subset_max = subset_df[subset_df.index <= peak_index.loc[tid, 'peak_id']].sum()
            means = means.append(pd.Series(subset_max))
        return means

    def get_captured_profit(self, average_win, average_unrealized_profit):
        return average_win/average_unrealized_profit

    def get_captured_loss(self, average_loss, average_unrealized_loss):
        return average_loss/average_unrealized_loss

    def green_red_candle_ratio(self, pnl_index):
        num_green_candles = len(pnl_index[pnl_index['pnl'] > 1])
        num_red_candles = len(pnl_index[pnl_index['pnl'] < 1])
        return num_green_candles/num_red_candles

    def get_minimum_rrr(self, longest_run, longest_drawdown):
        return round((longest_run - 1) / (1 - longest_drawdown), 5)

    def get_average_rrr(self, average_win, average_loss):
        return average_win/average_loss

    def get_unrealized_rrr(self, unrealized_profit, unrealized_loss):
        return unrealized_profit/unrealized_loss

    def get_amount_of_historical_data(self, df):
        return len(df)

    def get_trade_activity_ratio(self, pnl_index, df):
        return len(pnl_index)/len(df)


    def summarize_statistics(self, capital: float = 9083.0) -> str:
        self.initial_capital, self.capital = capital, capital
        """
        Summarizes all calculated statistics into a statement:

        :return str with the following form:

            Strategy statistics:
            Number of Trades:                  X
            Trades Won vs Trades Lost:         X
            Largest Profit vs Largest Loss:    X
            Win Streak vs Losing Streak:       X
            Largest Run vs Largest Drawdown:   X
            Profit vs Loss Per Trade:          X 
            Average Risk-Reward:               X
            Minimum Risk-Reward:               X
            Win Percentage:                    X
            Profit Factor:                     X

        """
        trades = self.get_trades()

        pnl = self.get_pnl(trades)

        initial_price = format(round(float(self.trade_history.first_trade().price), 2), ",")
        final_price = format(float(self.trade_history.last_trade().price), ",")
        initial_capital = format(self.initial_capital, ",")
        hold_new_capital = format(round(self.initial_capital * self.trade_history.last_trade().price /
                                        self.trade_history.first_trade().price, 2), ",")
        fab_new_capital = format(round(self.capital * pnl, 5), ",")

        average_win = self.get_average_win(self.get_gross_profit(trades), self.get_num_trades_won(trades))
        average_loss = self.get_average_loss(self.get_gross_loss(trades), self.get_num_trades_lost(trades))
        longest_run = self.get_longest_run(trades)
        longest_drawdown = self.get_longest_drawdown(trades)
        win_streak = self.get_win_streak(trades)
        lose_streak = self.get_lose_streak(trades)
        num_trades_lost = self.get_num_trades_lost(trades)
        num_trades_won = self.get_num_trades_won(trades)
        largest_profit = self.get_largest_profit(trades)
        largest_loss = self.get_largest_loss(trades)

        avg_rrr = round(((average_win - 1) * 100) / ((1 - average_loss) * 100), 5) if \
            average_win and average_loss != 1 else 0
        min_rrr = round((longest_run - 1) / (1 - longest_drawdown), 5) if \
            longest_run and longest_drawdown != 1 else 0

        statement = f""" 
        Data is provided from {self.trade_history.first_trade().datetime} to {self.trade_history.last_trade().datetime}
        Price of the Traded Asset went from {initial_price} to {final_price} 

        If you held {initial_capital} worth of the asset, it would be: {hold_new_capital} now
        However, using the FAB method, it would be: {fab_new_capital} now

        Strategy statistics:
            Number of Trades:                  {num_trades_lost + num_trades_won} 
            Trades Won vs Trades Lost:         {num_trades_won} vs {num_trades_lost} 
            Largest Profit vs Largest Loss:    {round(largest_profit, 5)} vs {round(largest_loss, 5)}
            Win Streak vs Losing Streak:       {win_streak} vs {lose_streak}
            Largest Run vs Largest Drawdown:   {round(longest_run, 5)} vs {round(longest_drawdown, 5)}
            Profit vs Loss Per Trade:          {round(average_win, 5)} vs {round(average_loss, 5)}
            Average Risk-Reward:               {avg_rrr}
            Minimum Risk-Reward:               {min_rrr}
            Win Percentage:                    {round(num_trades_won / (num_trades_lost + num_trades_won), 5)}
            Profit Factor:                     {round(pnl, 3)}x
        """
        return statement
