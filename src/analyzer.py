from trading_history import TradeHistory
from exceptions import Exceptions

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

    def get_num_trades_won(self, trades):
        return len([trade for trade in trades if trade > 1])

    def get_num_trades_lost(self, trades):
        return len([trade for trade in trades if trade <= 1])

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


    def calculate_statistics(self, trade_history: TradeHistory, commission= 0.9996) -> str:
        """
        Calculates the following metrics:
        Profit Factor, trades won, trades lost, gross profit, gross loss, largest profit, largest loss

        params: list of lists, one row having the form: [LONG/SHORT, ENTER/EXIT, DATETIME, PRICE, RULE #],
                Ex. ['Short', 'Enter', '2018-04-09 11:37:00', 6745.98, 'Rule 1'],

        :return None
        """
        # Initializing all variables
        self.trade_history = trade_history
        self.trades = []
        profitability = 1

        Exceptions.check_empty_trade_history(self.trade_history.allTrades)

        # Ensures all reported trades have been closed
        if self.trade_history.last_trade().status == "Enter":
            self.trade_history.allTrades = self.trade_history[:-1]

        for i in range(1, len(self.trade_history), 2):
            enter_trade = trade_history[i]
            exit_trade = trade_history[i+1]
            Exceptions.check_trade_status_exists(enter_trade)

            if enter_trade.side == "Short":
                profitability = self.calculate_short_profitability(enter_trade.price, exit_trade.price, commission)
            elif enter_trade.side == "Long":
                profitability = self.calculate_long_profitability(enter_trade.price, exit_trade.price, commission)

            # Final form of profitability is: 1 + profit margin. Profit margin CAN be negative here.
            self.trades.append(round(profitability, 6))

        return "calculated stats"

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
