import pandas as pd
import sqlalchemy
from backtester import Backtester
from fab_strategy import FabStrategy
from trader import Trader
from datetime import datetime, timedelta
from dataloader import _DataLoader
from IPython.display import display, clear_output
from playsound import playsound

class Screener:

    def __init__(self):
        self.loader = _DataLoader()
        self.master_screener = pd.DataFrame()

    def check_for_signals(self, df, strategy, max_candle_history=10):
        strategy.load_data(df)
        strategy.update_moving_averages()

        # list_of_enters = {'Rule1 Buy': strategy.rule_1_buy_enter,
        #                  'Rule1 Short': strategy.rule_1_short_enter,
        #                  'Rule2 Buy': strategy.rule_2_buy_enter,
        #                  'Rule2 Short':strategy.rule_2_short_enter,
        #                  'Rule3 Buy': strategy.rule_3_buy_enter,
        #                  'Rule3 Short': strategy.rule_3_short_enter}
        #
        # list_of_exits = {'Rule1 Buy': strategy.rule_1_buy_exit,
        #                  'Rule1 Short': strategy.rule_1_short_exit,
        #                  'Rule2 Buy': strategy.rule_2_buy_stop,
        #                  'Rule2 Short': strategy.rule_2_short_stop,
        #                  'Rule3 Buy': strategy.rule_1_buy_exit,
        #                  'Rule3 Short': strategy.rule_1_short_exit,
        #                  }

        list_of_enters = {
                         'Rule2 Buy': strategy.rule_2_buy_enter,
                         'Rule2 Short':strategy.rule_2_short_enter,
                        }

        list_of_exits = {
                         'Rule2 Buy': strategy.rule_2_buy_stop,
                         'Rule2 Short': strategy.rule_2_short_stop
                         }


        for rule in list_of_enters:
            for x_last_row in range(-10,0):
                if list_of_enters[rule](x_last_row) != True:
                    continue
                for remaining_row in range(x_last_row + 1, 0):
                    if list_of_exits[rule](remaining_row) == True:
                        return False, None, None
                return True, x_last_row, str(rule)
        return False, None, None

    def screen(self, trader, max_candle_history=10):
        cursor = self.loader.sql.conn.cursor()
        strategy = FabStrategy()
        """ Add 'side when setting index"""
        df_metrics = self.loader.sql.SELECT(f"* from metrics order by symbol", cursor).set_index(['symbol', 'tf', 'rule_no', 'side', 'metric_id']).astype(float)
        partial_screener = pd.DataFrame(columns=['date', 'signal', 'how recent', 'metric_id'])
        master_screener = partial_screener
        current_symbol, current_tf = None, None
        signal = False
        count = 0

        while True:
            count += 1
            for symbol, tf, rules, side, metric_id in df_metrics.index[:]:
                if tf in [7, 21,77, 1, 3, 5]:
                    continue
                if symbol == current_symbol and tf == current_tf:
                    continue

                if (symbol, tf) in master_screener.index:
                    if (symbol, tf) in self.master_screener.index:
                        now = datetime.now()
                        signal_date = self.master_screener.loc[(symbol, tf), 'date']
                        self.master_screener.loc[(symbol, tf), 'how recent'] = -int((now - signal_date).total_seconds()/(60*tf))
                        self.master_screener['most recent'] = master_screener.groupby('symbol')['how recent'].transform(
                            'max')
                        self.master_screener = self.master_screener.sort_values(
                                                            ['most recent', 'symbol', 'how recent', 'unrealized_rrr'],
                                                            ascending=[False, False, False, False])
                        if self.master_screener.loc[(symbol, tf), 'how recent'] <-10:
                            self.master_screener.drop((symbol, tf), inplace=True)
                    continue

                # print(f'checking: {symbol, tf} ')
                df = trader.set_asset(symbol, tf, max_candles_needed=231 + max_candle_history + 1, drop_last_row=False)

                signal, x_many_candles_ago, rule = self.check_for_signals(df, strategy, max_candle_history)
                current_symbol, current_tf = symbol, tf

                if signal:
                    date = datetime.now() - timedelta(minutes=abs(x_many_candles_ago)*tf)
                    partial_screener = partial_screener.append(pd.DataFrame([[date, rule, x_many_candles_ago, metric_id]],
                                                        columns=['date', 'signal', 'how recent', 'metric_id']))
                    signal, x_many_candles_ago= None, None
                    master_screener = partial_screener.merge(df_metrics.reset_index()
                                                                  [['metric_id', 'symbol', 'tf', 'unrealized_rrr',
                                                                    'peak_unrealized_profit', 'num_candles_to_peak',
                                                                    'amount_of_data']],
                                                                  on='metric_id')
                    master_screener['most recent'] = master_screener.groupby('symbol')['how recent'].transform('max')
                    master_screener = master_screener.set_index(['symbol', 'tf']).sort_values(
                        ['most recent', 'symbol', 'how recent', 'unrealized_rrr'], ascending=[False, False, False, False])
                    self.master_screener = master_screener[master_screener['amount_of_data'] >= 10].copy()

                    if count > 1 and (symbol, tf) in self.master_screener:
                        print(f'new value added: {symbol, tf}')
                        playsound(r"C:\Users\haseab\Desktop\Python\PycharmProjects\FAB\local\cash_register.mp3")

                    yield self.master_screener
            yield self.master_screener

        # return self.master_screener.set_index(['symbol','tf']).sort_values("unrealized_rrr", ascending=False)

    def screen_monitor(self, trader):
        screen_generator = self.screen(trader=trader)
        while True:
            clear_output(wait=True)
            try:
                display(next(screen_generator))
            except StopIteration:
                return self.master_screener






