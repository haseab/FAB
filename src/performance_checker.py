import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
sys.path.insert(1, r'/FAB/src')
from trader import Trader
from IPython.display import clear_output


class PerformanceChecker:

    def __init__(self, current_capital=None):
        self.trader = Trader(db=False)
        self.resp = self.trader.load_account()
        self.current_capital = current_capital
        self.live_capital = None
        self.goal_limit_amount = None

    def create_goal(self, goal_date, start_date=None):
        start_date = datetime.now() if not start_date else start_date
        current_capital = self.current_capital
        df_bounds = pd.DataFrame(columns=['date', 'week number', 'lower limit', 'goal limit', 'med limit', 'upper limit'])

        df_bounds['date'] = pd.date_range(start_date, goal_date, freq='7D')

        week_number = np.linspace(0, len(df_bounds)-1, len(df_bounds)).astype('int64')
        df_bounds['lower limit'] = current_capital * 1.005**week_number
        df_bounds['goal limit'] = current_capital * 1.015**week_number
        df_bounds['med limit'] = current_capital * 1.025**week_number
        df_bounds['upper limit'] = current_capital * 1.035**week_number
        df_bounds['week number'] = week_number
        df_bounds.to_csv('trading_goal_limits.csv', index=False)

        df_balances = pd.DataFrame()
        df_balances = df_balances.append(pd.DataFrame([[start_date, 0, current_capital]],
                                                      columns=['date', 'week_num', 'capital']))
        df_balances.to_csv('trading_actual_values.csv', index=False)

        goal_amount =df_bounds["goal limit"].iloc[-1]
        print(f'Current_money:      {self.current_capital:,}')
        print(f'Goal by {goal_date}: {goal_amount:,}')
        return df_bounds


    def update_progress(self, additional_balance=0, commit=False, datetime_adjust=None):
        datetime_adjust = timedelta(0) if not datetime_adjust else datetime_adjust
        seconds_in_week = 604_800
        df_balances = pd.read_csv('trading_balance.csv')
        df_balances['date'] = df_balances['date'].astype('datetime64')

        start_date = df_balances['date'].iloc[0].to_pydatetime()
        self.live_capital = self.trader.get_capital(additional_balance)
        live_date = datetime.now() + datetime_adjust
        week_num = round((live_date - start_date).total_seconds() / seconds_in_week, 3)

        df_balances = df_balances.append(pd.DataFrame([[live_date, week_num, self.live_capital]],
                                                      columns=['date', 'week_num', 'capital']))
        if commit:
            df_balances.to_csv('trading_balance.csv', index=False)

        self.check_progress(df_balances)
        return df_balances

    def check_progress(self, df_balances=None, close_up=False):
        df_bounds = pd.read_csv('trading_goal_limits.csv')
        df_bounds['date'] = df_bounds['date'].astype('datetime64')
        df_balances = pd.read_csv('trading_balance.csv') if type(df_balances) == type(None) else df_balances
        self.df_balances = df_balances
        x1 = df_bounds['week number']

        plt.figure(figsize=(20, 10))
        plt.plot(x1, df_bounds['lower limit'], color='red')
        plt.plot(x1, df_bounds['goal limit'], color='gray')
        plt.plot(x1, df_bounds['med limit'], color='gray')
        plt.plot(x1, df_bounds['upper limit'], color='lime')

        x2 = df_balances['week_num']
        y2 = df_balances['capital']

        if close_up:
            plt.xlim(0, 2*x2.iloc[-1]) if x2.iloc[-1] != 0 else plt.xlim(0, 1)
            plt.ylim(min(y2), max(y2))
        else:
            current_week = df_bounds[df_bounds['date'] <= datetime.now()]['week number'].iloc[-1] + 1
            green_line = df_bounds.set_index('week number').loc[1]['upper limit']
            y2_min = min(y2)

            plt.xlim(0, current_week)
            plt.ylim(min(19350, y2_min), green_line)
        plt.scatter(x2, y2, color='black')
        plt.plot(x2, y2, color='blue', linewidth=3)

        return plt.show()

    def monitor_progress(self):
        self.update_progress(5345, commit=False)
        while True:
            time.sleep(86400)
            clear_output(wait=True)
            self.update_progress(5345, commit=True)



