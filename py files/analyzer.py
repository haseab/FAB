from trading_history import TradeHistory


class Analyzer:
    """
    Responsible for analysis of trading history.

    Attributes
    -----------
    trade_history: list of lists
    trades: list of floats

    profit: float
    gross_profit: float
    gross_loss: float
    largest_profit: float
    largest_loss: float
    longest_run: float
    longest_drawdown: float
    average_win: float
    average_loss: float

    trades_won: int
    trades_lost: int
    win_streak: int
    lose_streak: int


    Methods
    ------------
    calculate_longest_run
    calculate_longest_drawdown
    calculate_statistics
    summarize_statistics

    Please look at each method for descriptions
    """

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

        return longest_drawdown, longest_counter

    def calculate_statistics(self, trade_history: TradeHistory) -> str:
        """
        Calculates the following metrics:
        Profit Factor, trades won, trades lost, gross profit, gross loss, largest profit, largest loss

        params: list of lists, one row having the form: [LONG/SHORT, ENTER/EXIT, DATETIME, PRICE, RULE #],
                Ex. ['Short', 'Enter', '2018-04-09 11:37:00', 6745.98, 'Rule 1'],

        :return None
        """
        # Initializing all variables
        self.trade_history = trade_history
        self.profit = 1
        self.trades_won, self.trades_lost = 0, 0
        self.gross_profit, self.gross_loss = 1, 1
        self.largest_profit, self.largest_loss = 1, 1
        self.trades = []

        if self.trade_history.allTrades == [['List of Trades']]:
            return "Not enough data to provide statistics"

        # Ensures all reported trades have been closed
        if self.trade_history.last_trade().status == "Enter":
            self.trade_history = self.trade_history[:-1]

        # Every set of 2 consecutive rows is an "Enter" and an "Exit" trade. Considered as one trade
        for i in range(1, len(self.trade_history), 2):

            one_trade = trade_history[i:i + 2]  # Length of 2
            if one_trade[0].status != "Enter" or one_trade[1].status != "Exit":
                raise Exception("Trading History doesn't have alternating Entering and Exit positions")

            if one_trade[0][0] == "Short":
                # 0.999 is considering commission costs. The rest is an equation for profitability
                profitability = 0.999 * (2 - one_trade[1][3] / one_trade[0][3])
            elif one_trade[0][0] == "Long":
                profitability = 0.999 * (one_trade[1][3] / one_trade[0][3])

            try:
                self.profit *= profitability
            except NameError:
                raise Exception("Something is wrong with TradeHistory. No Long/Short")

            # Final form of profitability is: 1 + profit margin. Profit margin CAN be negative here.
            self.trades.append(round(profitability, 4))

            if profitability > 1:
                self.trades_won += 1
                self.gross_profit *= profitability
            elif profitability < 1:
                self.trades_lost += 1
                self.gross_loss *= profitability
            self.largest_loss = min(profitability, self.largest_loss)
            self.largest_profit = max(profitability, self.largest_profit)

        self.longest_run, self.win_streak = self.calculate_longest_run(self.trades)
        self.longest_drawdown, self.lose_streak = self.calculate_longest_drawdown(self.trades)
        self.average_win = self.gross_profit ** (1 / self.trades_won)
        self.average_loss = self.gross_loss ** (1 / self.trades_lost)
        return "calculated stats"

    def summarize_statistics(self, capital: float = 50000.0) -> str:
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

        initial_price = format(round(float(self.trade_history.first_trade().price), 2), ",")
        final_price = format(float(self.trade_history.last_trade().price), ",")
        initial_capital = format(self.initial_capital, ",")
        hold_new_capital = format(round(self.initial_capital * self.trade_history.last_trade().price /
                                   self.trade_history.first_trade().price, 2), ",")
        fab_new_capital = format(round(self.capital * self.profit, 5), ",")


        statement = f""" 
        Data is provided from {self.trade_history.first_trade().datetime} to {self.trade_history.last_trade().datetime}
        Price of the Traded Asset went from {initial_price} to {final_price} 
        If you had {initial_capital}, normally it would have turned into: {hold_new_capital}
        However, using the FAB method, it would turn into: {fab_new_capital}

        Strategy statistics:
            Number of Trades:                  {self.trades_lost + self.trades_won} 
            Trades Won vs Trades Lost:         {self.trades_won} vs {self.trades_lost} 
            Largest Profit vs Largest Loss:    {round(self.largest_profit, 5)} vs {round(self.largest_loss, 5)}
            Win Streak vs Losing Streak:       {self.win_streak} vs {self.lose_streak}
            Largest Run vs Largest Drawdown:   {round(self.longest_run, 5)} vs {round(self.longest_drawdown, 5)}
            Profit vs Loss Per Trade:          {round(self.average_win, 5)} vs {round(self.average_loss, 5)}
            Average Risk-Reward:               {round(((self.average_win - 1) * 100) / ((1 - self.average_loss) * 100), 5)}
            Minimum Risk-Reward:               {round((self.longest_run - 1) / (1 - self.longest_drawdown), 5)}
            Win Percentage:                    {round(self.trades_won / (self.trades_lost + self.trades_won), 5)}
            Profit Factor:                     {round(self.profit, 3)}x
        """
        return statement
