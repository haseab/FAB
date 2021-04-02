import pandas as pd
import sqlalchemy
from backtester import Backtester
from fab_strategy import FabStrategy
from trader import Trader

class Screener():

    def __init__(self):
        self.engine = sqlalchemy.create_engine('mysql+pymysql://root:***REMOVED***@localhost:3306/sample')

    def check_for_signals(self, df, strategy):
        strategy.load_data(df)
        strategy.update_moving_averages()

        list_of_rules = [strategy.rule_1_buy_enter,
                         strategy.rule_1_short_enter,
                         strategy.rule_2_buy_enter,
                         strategy.rule_2_short_enter,
                         strategy.rule_3_buy_enter,
                         strategy.rule_3_short_enter]

        for rule in list_of_rules:
            for x_last_row in range(-10,0):
                if rule(x_last_row) == True:
                    signal_row = df.iloc[len(df) + x_last_row, :]
                    return True, x_last_row, str(rule)

        signal_row = df.iloc[len(df) -1, :]

        return False, None, None


    def screen(self):
        df_metrics = pd.read_sql("SELECT * FROM metrics", self.engine).drop("id", axis=1).set_index(["symbol", "timeframe"])

        screener_columns = ["Signal?", "How Recent?"] + list(df_metrics.columns)
        screener = pd.DataFrame(columns=screener_columns).set_index(["symbol", "timeframe"])

        for symbol, tf in df_metrics.index:
            df = pd.read_sql(f"SELECT * FROM candlesticks WHERE SYMBOL = {symbol} and TIMEFRAME = {tf}", self.engine)
            signal, x_many_minutes_ago, rule = self.check_for_signals(df, FabStrategy())

            if signal:
                screener.append(df.loc[symbol, tf])

                last_row = len(df) - 1
                screener.loc[last_row, 'Signal?'] = rule
                screener.loc[last_row, 'How Recent?'] = x_many_minutes_ago

        return screener.sort_values("RRR")






