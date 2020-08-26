class Backtester():
    def __init__(self, data, strategy):
        """
        Parameters:
            strategy: The trading strategy used for analysis
            Returns: list of strings, each string in the following format:\

            '{'Buy'/'Short'/'Buyclose'/'Shortclose'} on {date} at Price {price} -Rule#{number}'
        """
        self.dataframe = data
        self.strategy = strategy

    def strategy(self, start, end, tf, args):
        """INSERT YOUR STRATEGY HERE"""
        pass


    def trade_stats(self, list_of_trades):
        initial_capital, capital = 30000, 30000
        capitalbought, capitalsold = [], []
        trades_won, trades_lost = 0, 0
        gross_profit, gross_loss = 1, 1
        largest_profit, largest_loss = 1, 1
        average_win, average_loss = 1, 1
        trade_index = []
        run, drawdown = 1, 1
        run_counter, drawdown_counter = 0, 0
        max_run, max_drawdown = 1, 1

        if list_of_trades == []:
            print("Not enough data")

        slice = [list_of_trades[i][0].split() for i in range(len(list_of_trades))]

        if slice[0][0] != "Buyclose" or slice[0][0] != "Shortclose":
            for x in range(len(slice)):
                if "Buy" == slice[x][0] or "Shortclose" == slice[x][0]:
                    capitalbought.append((slice[x][6]))
                if "Short" == slice[x][0] or "Buyclose" == slice[x][0]:
                    capitalsold.append((slice[x][6]))

        if len(capitalbought) > len(capitalsold):
            capitalbought.pop()

        elif len(capitalbought) < len(capitalsold):
            capitalsold.pop()

        for i in range(len(capitalbought)):
            profitability = float(capitalsold[i]) / float(capitalbought[i])
            capital *= profitability
            trade_index.append(profitability)
            # print(profitability)
            if profitability > 1:
                gross_profit *= profitability
            elif profitability < 1:
                gross_loss *= profitability
            if profitability > largest_profit:
                largest_profit = profitability
            elif profitability < largest_loss:
                largest_loss = profitability
            if capitalbought[i] > capitalsold[i]:
                trades_lost += 1
            elif capitalsold[i] > capitalbought[i]:
                trades_won += 1
                # print(trade_index)

        for i in range(len(trade_index)):
            if trade_index[i] < 1:
                if drawdown_counter == 0:
                    if drawdown != 1:
                        if max_drawdown > drawdown:
                            max_drawdown = drawdown
                        drawdown = 1
                    run_counter = 0
                    drawdown_counter = trade_index[i]
                    drawdown *= trade_index[i]
                if drawdown_counter == trade_index[i - 1]:
                    drawdown_counter = trade_index[i]
                    drawdown *= trade_index[i]
                if max_drawdown > drawdown:
                    max_drawdown = drawdown
            if trade_index[i] > 1:
                if run_counter == 0:
                    if run != 1:
                        if max_run < run:
                            max_run = run
                        run = 1
                    drawdown_counter = 0
                    run_counter = trade_index[i]
                    run *= trade_index[i]
                if run_counter == trade_index[i - 1]:
                    run_counter = trade_index[i]
                    run *= trade_index[i]
                if max_run < run:
                    max_run = run

        average_win = gross_profit ** (1 / trades_won)
        average_loss = gross_loss ** (1 / trades_lost)

        statement = f"""
        Price of asset went from {format(round(float(capitalbought[0]), 2), ",")} to {format(float(capitalsold[-1]), ",")} 

        Strategy statistics:
            Number of Trades: {trades_lost + trades_won} 
            Number of Trades Won - Trades Lost: {trades_won} - {trades_lost} 
            Largest Single Profit - Largest Single Loss: {round(largest_profit, 5)} - {round(largest_loss, 5)}
            Largest Run - Largest Drawdown: {round(max_run, 5)} - {round(max_drawdown, 5)}
            Profit - Loss Per Trade: {round(average_win, 5)} - {round(average_loss, 5)}
            Risk-Reward: {round(((average_win - 1) * 100) / ((1 - average_loss) * 100), 5)}
            Win Percentage: {round(trades_won / (trades_lost + trades_won), 5)}
            Profit Factor: {round(capital / initial_capital, 3)}x
        """
        return statement