class StrategyExecutor():
    #     def __init__(self, data, strategy):
    #         """
    #         Parameters:
    #             strategy: The trading strategy used for analysis
    #             Returns: list of strings, each string in the following format:\

    #             '{'Buy'/'Short'/'Buyclose'/'Shortclose'} on {date} at Price {price} -Rule#{number}'
    #         """
    #         self.dataframe = data
    #         self.strategy = strategy

    def fab_strategy(self, df, size=7, size2=77, size3=231, size4=880, size5=2240):
        def sma(dataframe, size):
            return round(dataframe['Close'].rolling(size).mean(), 2)

        date = [str(df['Datetime'].iloc[i]) for i in range(len(df['Datetime']))]
        price, low, high = df['Close'].values, df['Low'].values, df['High'].values
        green, orange, black, blue, pink = sma(df, size).values, sma(df, size2).values, sma(df, size3).values, sma(df,
                                                                                                                   size4).values, sma(
            df, size5).values
        trade_history = ['List of Trades']
        num_of_trades = 0

        for i in range(231, len(green) - 1):
            ### Rule 1
            if green[i] > black[i] and orange[i] > black[i]:
                if green[i] <= orange[i]:
                    if green[i + 1] > orange[i + 1] and trade_history[-1][1] != "Enter":
                        trade_history.append(["Long", "Enter", date[i + 1], price[i + 1], "Rule 1"])
                elif green[i] >= orange[i] and trade_history[-1][:2] == ["Long", 'Enter']:
                    if green[i + 1] < orange[i + 1] or orange[i + 1] < black[i + 1]:
                        trade_history.append(["Long", "Exit", date[i + 1], price[i + 1], "Rule 1"])

            elif green[i] < black[i] and orange[i] < black[i]:
                if green[i] >= orange[i]:
                    if green[i + 1] < orange[i + 1] and trade_history[-1][1] != "Enter":
                        trade_history.append(["Short", "Enter", date[i + 1], price[i + 1], "Rule 1"])
                elif green[i] <= orange[i] and trade_history[-1][:2] == ["Short", 'Enter']:
                    if green[i + 1] > orange[i + 1] or orange[i + 1] > black[i + 1]:
                        trade_history.append(["Short", "Exit", date[i + 1], price[i + 1], "Rule 1"])

                        ## Rule 2
            if low[i] > black[i] and low[i - 1] > black[i - 1] and green[i] > black[i] and orange[i] <= black[i] and \
                    trade_history[-1][1] != "Enter":
                if low[i + 1] <= black[i + 1] and ((orange[i] - orange[i - 3]) / 3) > (
                        (black[i] - black[i - 3]) / 3):  # If slope of orange MA is greater than black MA
                    trade_history.append(["Long", "Enter", date[i + 1], black[i + 1], "Rule 2"])
            elif price[i] < black[i] and orange[i] <= black[i] and trade_history[-1][:2] == ["Long", 'Enter'] and \
                    trade_history[-1][-1] == "Rule 2":
                if price[i + 1] <= orange[i + 1] or green[i + 1] < black[i + 1]:
                    trade_history.append(["Long", "Exit", date[i + 1], price[i + 1], "Rule 2"])

            if high[i] < black[i] and high[i - 1] < black[i - 1] and green[i] < black[i] and orange[i] >= black[i] and \
                    trade_history[-1][1] != "Enter":
                if high[i + 1] >= black[i + 1] and ((orange[i] - orange[i - 3]) / 3) < (
                        (black[i] - black[i - 3]) / 3):  # If slope of orange MA is less than black MA
                    trade_history.append(["Short", "Enter", date[i + 1], black[i + 1], "Rule 2"])
            elif price[i] > black[i] and orange[i] >= black[i] and trade_history[-1][:2] == ["Short", 'Enter'] and \
                    trade_history[-1][-1] == "Rule 2":
                if price[i + 1] >= orange[i + 1] or green[i + 1] > black[i + 1]:
                    trade_history.append(["Short", "Exit", date[i + 1], price[i + 1], "Rule 2"])

            ### Rule 3
            if green[i] > black[i] and orange[i] <= black[i] and trade_history[-1][1] != "Enter":
                if orange[i + 1] > black[i + 1] and green[i + 1] > orange[i + 1]:
                    trade_history.append(["Long", "Enter", date[i + 1], price[i + 1], "Rule 3"])
            elif green[i] < black[i] and orange[i] >= black[i]:
                if orange[i + 1] < black[i + 1] and green[i + 1] < orange[i + 1] and trade_history[-1][1] != "Enter":
                    trade_history.append(["Short", "Enter", date[i + 1], price[i + 1], "Rule 3"])

        return trade_history




