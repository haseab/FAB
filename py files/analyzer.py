class Analyzer():
    #     def __init__(self, tradehistory):
    #         self.history =

    def calculate_longest_run(self, trade_index):
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

    def calculate_longest_drawdown(self, trade_index):
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

    def calculate_statistics(self, trade_history):
        self.trade_history = trade_history
        self.initial_capital, self.capital = 30000, 30000
        self.trades_won, self.trades_lost = 0, 0
        self.gross_profit, self.gross_loss = 1, 1
        self.largest_profit, self.largest_loss = 1, 1
        self.trades = []

        if self.trade_history == []:
            return "Not enough data to provide statistics"
        if self.trade_history[-1][1] == "Enter":
            self.trade_history = self.trade_history[:-1]

        for i in range(1, len(self.trade_history), 2):
            one_trade = trade_history[i:i + 2]
            if one_trade[0][0] == "Short":
                profitability = 2 - one_trade[1][3] / one_trade[0][3]
            elif one_trade[0][0] == "Long":
                profitability = one_trade[1][3] / one_trade[0][3]
            self.capital *= profitability
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
        return

    def summarize_statistics(self):
        statement = f""" 
        Price of the Traded Asset went from {format(round(float(self.trade_history[1][3]), 2), ",")} to {format(float(self.trade_history[-1][3]), ",")} 
        If you had {format(self.initial_capital, ",")}, normally it would have turned into: {format(round(self.initial_capital * self.trade_history[-1][3] / self.trade_history[1][3], 2), ",")}
        However, using the FAB method, it would turn into: {format(round(self.capital, 5), ",")}

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
            Profit Factor:                     {round(self.capital / self.initial_capital, 3)}x
        """
        return statement
