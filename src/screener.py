import pandas as pd
import sqlalchemy
from backtester import Backtester
from fab_strategy import FabStrategy
from trader import Trader
import time
from datetime import datetime, timedelta
from dataloader import _DataLoader
from IPython.display import display, clear_output
from playsound import playsound
from helper import Helper
from concurrent.futures import ThreadPoolExecutor

class Screener:

    def __init__(self):
        self.loader = _DataLoader()
        self.master_screener = pd.DataFrame()
        self.strategy = FabStrategy()

    def check_for_signals(self, df, strategy, max_candle_history=10):
        strategy.load_data(df)
        strategy.update_moving_averages()

        list_of_enters = {
                            ('Rule 1', 'Long'): strategy.rule_1_buy_enter,
                            ('Rule 1', 'Short'): strategy.rule_1_short_enter,
                            ('Rule 2', 'Long'):  strategy.rule_2_buy_enter,
                            ('Rule 2', 'Short'): strategy.rule_2_short_enter,
                            ('Rule 3', 'Long'): strategy.rule_3_buy_enter,
                            ('Rule 3', 'Short'): strategy.rule_3_short_enter
                          }

        list_of_exits = {
                            ('Rule 1', 'Long'): strategy.rule_1_buy_exit,
                            ('Rule 1', 'Short'): strategy.rule_1_short_exit,
                            ('Rule 2', 'Long'):  strategy.rule_2_buy_stop,
                            ('Rule 2', 'Short'): strategy.rule_2_short_stop,
                            ('Rule 3', 'Long'): strategy.rule_1_buy_exit,
                            ('Rule 3', 'Short'): strategy.rule_1_short_exit
                          }

        # list_of_enters = {
        #                 ('Rule 2', 'Long'): strategy.rule_2_buy_enter,
        #                 ('Rule 2', 'Short'): strategy.rule_2_short_enter,
        #                 }
        #
        # list_of_exits = {
        #                  ('Rule 2', 'Long'): strategy.rule_2_buy_stop,
        #                  ('Rule 2', 'Short'): strategy.rule_2_short_stop
        #                  }
        #

        for rule, side in list_of_enters:
            for x_last_row in range(-10,0):
                if list_of_enters[(rule,side)](x_last_row) != True:
                    continue
                for remaining_row in range(x_last_row + 1, 0):
                    if list_of_exits[(rule, side)](remaining_row) == True:
                        return False, None, None, None
                return True, x_last_row, rule, side
        return False, None, None, None

    def screen(self, trader, metrics_table, max_candle_history=10):
        metrics_table = metrics_table[:]
        symbols = pd.DataFrame(metrics_table.index.unique(level=0), columns=['symbol'])
        tfs = pd.DataFrame(metrics_table.index.unique(level=1), columns=['tf'])
        tfs = tfs.set_index('tf').drop([7, 21, 77, 240, 60]).reset_index()
        df_symbol_tf = symbols.merge(tfs, how='cross')
        max_candles_needed = [231+max_candle_history+1 for _ in range(len(df_symbol_tf))]

        # return df_symbol_tf
        partial_screener = pd.DataFrame(columns=['date', 'signal', 'how recent', 'metric_id'])

        # Grabs Binance data for every single (symbol, tf) pair
        with ThreadPoolExecutor(max_workers=50) as executor:
            results = executor.map(trader.set_asset, df_symbol_tf['symbol'].values, df_symbol_tf['tf'].values, max_candles_needed)
            executor.shutdown(wait=True)

        # return results

        for df, symbol, tf in zip(results, df_symbol_tf['symbol'], df_symbol_tf['tf']):
            signal, x_many_candles_ago, rule, side = self.check_for_signals(df, self.strategy, max_candle_history)

            if signal:
                # tf = int((df['date'].iloc[1]-df['date'].iloc[0]).total_seconds()/60)
                metric_id = metrics_table.loc[(symbol, tf, rule, side), 'metric_id']
                date = datetime.now() - timedelta(minutes=int(abs(x_many_candles_ago)*tf))
                change_factor = float(df['close'].iloc[-1]/df['close'].iloc[x_many_candles_ago])
                percentage_change = Helper.factor_to_percentage([change_factor])[0]
                partial_screener = partial_screener.append(pd.DataFrame([[metric_id, date, str((rule, side)), x_many_candles_ago, percentage_change]],
                                                    columns=['metric_id', 'date', 'signal', 'how recent', "% change"]))

        master_screener = partial_screener.merge(metrics_table.reset_index()
                                                      [['metric_id', 'symbol', 'tf', 'average_win', 'unrealized_rrr',
                                                        'peak_unrealized_profit', 'num_candles_to_peak',
                                                        'amount_of_data']],
                                                      on='metric_id')

        master_screener['most recent'] = master_screener.groupby('symbol')['how recent'].transform('max')
        master_screener = master_screener.set_index(['symbol', 'tf']).sort_values(
            ['most recent', 'symbol', 'how recent', 'unrealized_rrr'], ascending=[False, False, False, False])
        master_screener = master_screener[['metric_id'] + list(master_screener.columns.drop('metric_id'))]
        return master_screener

    def screen_monitor(self, trader, minutes_to_sleep=1):
        cursor = self.loader.sql.conn.cursor()
        df_metrics = self.loader.sql.SELECT(f"* from metrics order by symbol", cursor).set_index(
            ['symbol', 'tf', 'rule_no', 'side']).astype(float)
        count = 0

        now = datetime.now().minute
        while True:
            if datetime.now().minute == now + 1:
                count += 1
                print(datetime.now())
                now = datetime.now().minute
                master_screener = self.screen(trader=trader, metrics_table=df_metrics)
                clear_output(wait=True)
                if count > 1 and len(master_screener) != len(self.master_screener):
                    new_values = []
                    temp = pd.concat([master_screener, self.master_screener]).drop_duplicates(subset='metric_id', keep=False)
                    for symbol, tf in temp.index:
                        try:
                            if temp.loc[(symbol, tf), 'how recent'] == -1:
                                new_values.append((symbol, tf))
                        except Exception():
                            print(temp, symbol, tf)
                    if len(new_values) > 0:
                        # playsound(r"C:\Users\haseab\Desktop\Python\PycharmProjects\FAB\local\cash_register.mp3")
                        print("new values added: ")
                        print(new_values)

                self.master_screener = master_screener[master_screener['amount_of_data'] >= 5].copy()

                display(self.master_screener)

                time.sleep(50*minutes_to_sleep)
            time.sleep(0.5)







