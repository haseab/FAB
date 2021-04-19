import pandas as pd
import sqlalchemy
from backtester import Backtester
from fab_strategy import FabStrategy
from trader import Trader
from datetime import datetime
from dataloader import _DataLoader
from IPython.display import display, clear_output

class Screener:

    def __init__(self):
        self.loader = _DataLoader()

    def check_for_signals(self, df, strategy):
        strategy.load_data(df)
        strategy.update_moving_averages()

        list_of_rules = {'Rule1 Buy': strategy.rule_1_buy_enter,
                         'Rule1 Short': strategy.rule_1_short_enter,
                         'Rule2 Buy': strategy.rule_2_buy_enter,
                         'Rule2 Short':strategy.rule_2_short_enter,
                         'Rule3 Buy': strategy.rule_3_buy_enter,
                         'Rule3 Short': strategy.rule_3_short_enter}

        for rule in list_of_rules:
            for x_last_row in range(-10,0):
                if list_of_rules[rule](x_last_row) == True:
                    signal_row = df.iloc[len(df) + x_last_row, :]
                    return True, x_last_row, str(rule)

        signal_row = df.iloc[len(df) -1, :]

        return False, None, None


    def screen(self, trader):
        cursor = self.loader.sql.conn.cursor()

        """ Add 'side when setting index"""
        df_metrics = self.loader.sql.SELECT(f"* FROM metrics", cursor).set_index(['symbol', 'tf', 'rule_no', 'side', 'metric_id']).astype(float)
        partial_screener = pd.DataFrame(columns=['date', 'signal', 'how recent', 'metric_id'])
        current_symbol, current_tf = None, None
        signal = False

        for symbol, tf, rules, side, metric_id in df_metrics.index[:]:
            if tf in [7, 21,77]:
                continue
            if symbol == current_symbol and tf == current_tf:
                continue

            # print(f'checking: {symbol, tf} ')
            df = trader.set_asset(symbol, tf, max_candles_needed=242)

            signal, x_many_candles_ago, rule = self.check_for_signals(df, FabStrategy())
            current_symbol, current_tf = symbol, tf


            if signal:
                # print('SIGNAL!!!')
                date = str(datetime.now())[:19]
                partial_screener = partial_screener.append(pd.DataFrame([[date, rule, f"{x_many_candles_ago} candles ago", metric_id]],
                                                    columns=['date', 'signal', 'how recent', 'metric_id']))
                signal, x_many_candles_ago= None, None
                self.master_screener = partial_screener.merge(df_metrics.reset_index()
                                                              [['metric_id', 'symbol', 'tf', 'unrealized_rrr',
                                                                'peak_unrealized_profit', 'num_candles_to_peak']],
                                                              on='metric_id')

                yield self.master_screener.set_index(['symbol','tf']).sort_values("unrealized_rrr", ascending=False)


        # return self.master_screener.set_index(['symbol','tf']).sort_values("unrealized_rrr", ascending=False)


    def screen_monitor(self, trader):
        screen_generator = self.screen(trader=trader)
        while True:
            clear_output(wait=True)
            try:
                display(next(screen_generator))
            except StopIteration:
                return self.master_screener.set_index(['symbol','tf']).sort_values("unrealized_rrr", ascending=False)





