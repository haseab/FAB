import pandas as pd
import sqlalchemy
from backtester import Backtester
from fab_strategy import FabStrategy
from trader import Trader
from datetime import datetime
from dataloader import _DataLoader

class Screener:

    def __init__(self):
        self.loader = _DataLoader()

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


    def screen(self, trader):
        cursor = self.loader.sql.conn.cursor()

        """ Add 'side when setting index"""
        df_metrics = self.loader.sql.SELECT(f"* FROM metrics", cursor).set_index(['symbol', 'tf', 'rule_no','metric_id']).astype(float)
        partial_screener = pd.DataFrame(columns=['date', 'signal', 'how recent', 'metric_id'])
        current_symbol, current_tf = None, None
        signal = False

        for symbol, tf, rules, metric_id in df_metrics.index:
            if symbol == current_symbol and tf == current_tf:
                continue

            print(f'checking: {symbol, tf} ')
            df = trader.set_asset(symbol, tf, max_candles_needed=255)

            signal, x_many_minutes_ago, rule = self.check_for_signals(df, FabStrategy())
            current_symbol, current_tf = symbol, tf

            if signal:
                print('SIGNAL!!!')
                date = str(datetime.now())[:10]
                partial_screener.append(pd.DataFrame([[date, rule, x_many_minutes_ago, metric_id]],
                                                    columns=['date', 'signal', 'how recent', 'metric_id']))
                signal, x_many_minutes_ago= None, None

        master_screener = partial_screener.merge(df_metrics, on='metric_id')

        return master_screener.sort_values("unrealized_rrr", ascending=False)






