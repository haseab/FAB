def bm_strategy(instance, dataframe, size, size2, size3, size4, size5, tf, start, end):
    def sma(dataframe, size):
        return round(dataframe['Close'].rolling(size).mean(), 2)

    dfraw = instance.get_range(dataframe, start, end)
    df = instance.timeframe_setter(dfraw, tf)
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