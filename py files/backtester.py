class Backtester():
    def __init__(self, data, strategy, start, end, tf):
        """
        Parameters:
            strategy: The trading strategy used for analysis
            Returns: list of strings, each string in the following format:\

            '{'Buy'/'Short'/'Buyclose'/'Shortclose'} on {date} at Price {price} -Rule#{number}'
        """
        self.dataframe = data
        self.strategy = strategy
        self.start = start
        self.end = end
        self.timeframe = tf

    def strategy(start, end, tf, args):
        """INSERT YOUR STRATEGY HERE"""
        pass

    def bm_strategy(dataframe, size, size2, size3, size4, size5, tf, start, end):
        def sma(dataframe, size):
            return round(dataframe['Close'].rolling(size).mean(), 2)

        dfraw = load1.get_range(data, start, end)
        df = load1.timeframe_setter(dfraw, tf)
        date = [str(df['Datetime'].iloc[i]) for i in range(len(df['Datetime']))]
        price, low, high = df['Close'].values, df['Low'].values, df['High'].values
        green, orange, black, blue, pink = sma(df, size).values, sma(df, size2).values, sma(df, size3).values, sma(df,
                                                                                                                   size4).values, sma(
            df, size5).values

        trade_history = []
        num_of_trades = 0

        for i in range(231, len(green) - 1):
            ####Rule 5
            if black[i] > pink[i] and orange[i] < black[i] and black[i] < blue[i] and pink[i] != 0 and black[i] / pink[
                i] > 1.23:
                if price[i] > pink[i] and green[i] < orange[i]:
                    if low[i + 1] <= pink[i + 1] and trade_history[-1][0][6] == "o":
                        trade_history.append([f"Shortclose on {date[i + 1]} at Price: {pink[i + 1]} -Rule5"])
                        trade_history.append([f"Buy on {date[i + 1]} at Price: {pink[i + 1]} -Rule5"])
                    if green[i + 1] <= pink[i + 1] and trade_history[-1][0][-1] == "5":
                        trade_history.append([f"Buyclose on {date[i + 1]} at Price: {price[i + 1]} -Rule5L"])
                if green[i] > orange[i] and price[i] < black[i]:
                    if high[i + 1] >= black[i + 1] and trade_history != [] and trade_history[-1][0][-1] == "5":
                        trade_history.append([f"Buyclose on {date[i + 1]} at Price: {black[i + 1]} -Rule5W"])
                    if green[i + 1] < orange[i + 1] and trade_history != [] and trade_history[-1][0][-1] == "5":
                        trade_history.append([f"Buyclose on {date[i + 1]} at Price: {price[i + 1]} -Rule5L"])

            if black[i] < pink[i] and orange[i] > black[i] and black[i] > blue[i] and black[i] != 0 and pink[i] / black[
                i] > 1.23:
                if price[i] < pink[i] and green[i] > orange[i]:
                    if high[i + 1] >= pink[i + 1] and trade_history[-1][0][4] == "o":
                        trade_history.append([f"Buyclose on {date[i + 1]} at Price: {pink[i + 1]} -Rule5"])
                        trade_history.append([f"Short on {date[i + 1]} at Price: {pink[i + 1]} -Rule5"])
                    if green[i + 1] >= pink[i + 1] and trade_history[-1][0][-1] == "5":
                        trade_history.append([f"Shortclose on {date[i + 1]} at Price: {price[i + 1]} -Rule5L"])
                if green[i] < orange[i] and price[i] > black[i]:
                    if low[i + 1] <= black[i + 1] and trade_history != [] and trade_history[-1][0][6] == "5":
                        trade_history.append([f"Shortclose on {date[i + 1]} at Price: {black[i + 1]} -Rule5W"])
                    if green[i + 1] > orange[i + 1] and trade_history != [] and trade_history[-1][0][-1] == "5":
                        trade_history.append([f"Buyclose on {date[i + 1]} at Price: {price[i + 1]} -Rule5L"])

                        ###Rule 4
            if black[i] > blue[i] and blue[i] != 0 and black[i] / blue[i] > 1.23:
                if price[i] > blue[i] and green[i] > blue[i] and blue[i] != 0 and orange[i] / blue[i] > 1.085:
                    if low[i + 1] <= blue[i + 1] and trade_history[-1][0][6] == "o":
                        trade_history.append([f"Shortclose on {date[i + 1]} at Price: {blue[i + 1]} -Rule4S"])
                        trade_history.append([f"Buy on {date[i + 1]} at Price: {blue[i + 1]} -Rule4S"])
                    if green[i + 1] <= blue[i + 1] and trade_history[-1][0][-1] == "S":
                        trade_history.append([f"Buyclose on {date[i + 1]} at Price: {price[i + 1]} -Rule5L"])
                if green[i] > orange[i] and price[i] < black[i]:
                    if high[i + 1] >= black[i + 1] and trade_history != [] and trade_history[-1][0][-1] == "S":
                        trade_history.append([f"Buyclose on {date[i + 1]} at Price: {black[i + 1]} -Rule4SW"])
                    if green[i + 1] < orange[i + 1] and trade_history != [] and trade_history[-1][0][-1] == "S":
                        trade_history.append([f"Buyclose on {date[i + 1]} at Price: {price[i + 1]} -Rule4SL"])

            if black[i] < blue[i] and black[i] != 0 and blue[i] / black[i] > 1.23:
                if price[i] < blue[i] and green[i] < blue[i] and blue[i] != 0 and blue[i] / orange[i] > 1.085:
                    if high[i + 1] >= blue[i + 1] and trade_history[-1][0][4] == "o":
                        trade_history.append([f"Buyclose on {date[i + 1]} at Price: {blue[i + 1]} -Rule4S"])
                        trade_history.append([f"Short on {date[i + 1]} at Price: {blue[i + 1]} -Rule4S"])
                    if green[i + 1] >= blue[i + 1] and trade_history[-1][0][-1] == "S":
                        trade_history.append([f"Shortclose on {date[i + 1]} at Price: {price[i + 1]} -Rule5L"])
                if green[i] < orange[i] and price[i] > black[i]:
                    if low[i + 1] <= black[i + 1] and trade_history != [] and trade_history[-1][0][6] == "S":
                        trade_history.append([f"Shortclose on {date[i + 1]} at Price: {black[i + 1]} -Rule4SW"])
                    if green[i + 1] > orange[i + 1] and trade_history != [] and trade_history[-1][0][-1] == "S":
                        trade_history.append([f"Buyclose on {date[i + 1]} at Price: {price[i + 1]} -Rule4SL"])

            if black[i] > blue[i] and orange[i] < black[i] and blue[i] != 0 and black[i] / blue[i] > 1.10:
                if price[i] > blue[i] and green[i] > blue[i] and blue[i] != 0 and orange[i] / blue[i] > 1.085:
                    if low[i + 1] <= blue[i + 1] and trade_history[-1][0][6] == "o":
                        trade_history.append([f"Shortclose on {date[i + 1]} at Price: {blue[i + 1]} -Rule4"])

            if black[i] < blue[i] and orange[i] < black[i] and black[i] != 0 and blue[i] / black[i] > 1.10:
                if price[i] < blue[i] and green[i] < blue[i] and black[i] != 0 and blue[i] / orange[i] > 1.085:
                    if high[i + 1] >= blue[i + 1] and trade_history[-1][0][4] == "o":
                        trade_history.append([f"Buyclose on {date[i + 1]} at Price: {blue[i + 1]} -Rule4"])

                        ### Rule 1
            if green[i] > black[i] and orange[i] > black[i]:
                if green[i] <= orange[i]:
                    if green[i + 1] > orange[i + 1] and trade_history == []:
                        trade_history.append([f"Buy on {date[i + 1]} at Price: {price[i + 1]} -Rule1"])
                    if green[i + 1] > orange[i + 1] and trade_history != [] and trade_history[-1][0][6] != 'o' and \
                            trade_history[-1][0][4] != 'o':
                        trade_history.append([f"Buy on {date[i + 1]} at Price: {price[i + 1]} -Rule1"])
                if green[i] >= orange[i] and trade_history != [] and trade_history[-1][0][4] == 'o':
                    if green[i + 1] < orange[i + 1] or orange[i + 1] < black[i + 1]:
                        num_of_trades += 1
                        trade_history.append([f"Buyclose on {date[i + 1]} at Price: {price[i + 1]} -Rule1"])

            if green[i] < black[i] and orange[i] < black[i]:
                if green[i] >= orange[i]:
                    if green[i + 1] < orange[i + 1] and trade_history == []:
                        trade_history.append([f"Short on {date[i + 1]} at Price: {price[i + 1]} -Rule1"])
                    if green[i + 1] < orange[i + 1] and trade_history != [] and trade_history[-1][0][6] != 'o' and \
                            trade_history[-1][0][4] != 'o':
                        trade_history.append([f"Short on {date[i + 1]} at Price: {price[i + 1]} -Rule1"])
                if green[i] <= orange[i] and trade_history != [] and trade_history[-1][0][6] == 'o':
                    if green[i + 1] > orange[i + 1] or orange[i + 1] > black[i + 1]:
                        num_of_trades += 1
                        trade_history.append([f"Shortclose on {date[i + 1]} at Price: {price[i + 1]} -Rule1"])

                        ## Rule 2
            if price[i] > black[i] and price[i - 1] > black[i - 1] and price[i - 2] > black[i - 2] and orange[i] <= \
                    black[i] and trade_history == []:
                if low[i + 1] <= black[i + 1] and ((orange[i] - orange[i - 3]) / 3) > ((black[i] - black[i - 3]) / 3):
                    trade_history.append([f"Buy on {date[i + 1]} at Price: {black[i + 1]} -Rule2N"])
            if price[i] > black[i] and price[i - 1] > black[i - 1] and price[i - 2] > black[i - 2] and orange[i] <= \
                    black[i]:
                if low[i + 1] <= black[i + 1] and ((orange[i] - orange[i - 3]) / 3) > (
                        (black[i] - black[i - 3]) / 3) and trade_history != [] and trade_history[-1][0][4] != 'o' and \
                        trade_history[-1][0][6] != 'o':
                    trade_history.append([f"Buy on {date[i + 1]} at Price: {black[i + 1]} -Rule2N"])
            if price[i] < black[i] and orange[i] <= black[i]:
                if low[i + 1] <= orange[i + 1] and trade_history != [] and trade_history[-1][0][4] == 'o' and \
                        trade_history[-1][0][-1] == "N":
                    trade_history.append([f"Buyclose on {date[i + 1]} at Price: {orange[i + 1]} -Rule2N"])

            if price[i] < black[i] and price[i - 1] < black[i - 1] and price[i - 2] < black[i - 2] and orange[i] >= \
                    black[i] and trade_history == []:
                if high[i + 1] >= black[i + 1] and ((orange[i] - orange[i - 3]) / 3) < ((black[i] - black[i - 3]) / 3):
                    trade_history.append([f"Short on {date[i + 1]} at Price: {black[i + 1]} -Rule2N"])
            if price[i] < black[i] and price[i - 1] < black[i - 1] and price[i - 2] < black[i - 2] and orange[i] >= \
                    black[i]:
                if high[i + 1] >= black[i + 1] and ((orange[i] - orange[i - 3]) / 3) < (
                        (black[i] - black[i - 3]) / 3) and trade_history != [] and trade_history[-1][0][6] != 'o' and \
                        trade_history[-1][0][4] != 'o':
                    trade_history.append([f"Short on {date[i + 1]} at Price: {black[i + 1]} -Rule2N"])
            if price[i] > black[i] and orange[i] >= black[i]:
                if high[i + 1] >= orange[i + 1] and trade_history != [] and trade_history[-1][0][6] == 'o' and \
                        trade_history[-1][0][-1] == "N":
                    trade_history.append([f"Shortclose on {date[i + 1]} at Price: {orange[i + 1]} -Rule2N"])

            ### Rule 3
            if green[i] > black[i] and orange[i] <= black[i] and trade_history == []:
                if orange[i + 1] > black[i + 1] and green[i + 1] > orange[i + 1]:
                    trade_history.append([f"Buy on {date[i + 1]} at Price: {price[i + 1]} -Rule3"])
            if green[i] > black[i] and orange[i] <= black[i]:
                if orange[i + 1] > black[i + 1] and green[i + 1] > orange[i + 1] and trade_history != [] and \
                        trade_history[-1][0][4] != 'o' and trade_history[-1][0][6] != 'o':
                    trade_history.append([f"Buy on {date[i + 1]} at Price: {price[i + 1]} -Rule3"])

            if green[i] < black[i] and orange[i] >= black[i] and trade_history == []:
                if orange[i + 1] < black[i + 1] and green[i + 1] < orange[i + 1]:
                    trade_history.append([f"Short on {date[i + 1]} at Price: {price[i + 1]} -Rule3"])
            if green[i] < black[i] and orange[i] >= black[i]:
                if orange[i + 1] < black[i + 1] and green[i + 1] < orange[i + 1] and trade_history != [] and \
                        trade_history[-1][0][6] != 'o' and trade_history[-1][0][4] != 'o':
                    trade_history.append([f"Short on {date[i + 1]} at Price: {price[i + 1]} -Rule3"])

        return trade_history

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