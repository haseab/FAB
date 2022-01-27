import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go

from fab_strategy import FabStrategy
from helper import Helper


class Illustrator:
    """
    WORK IN PROGRESS - Responsible for analysis of trading history.

    Attributes
    -----------

    Methods
    ------------
    graph_data

    Please look at each method for descriptions
    """
    def __init__(self):
        self.strategy = FabStrategy()
        pass

    def set_graph_style(self, df_graph, sma=True, extra_mas=False):
        addplot = []
        haseab_colors = mpf.make_marketcolors(up='#4ea672', down='#cf4532', edge='black')
        self.haseab_style = mpf.make_mpf_style(marketcolors=haseab_colors, gridcolor='#d4d4d4')

        if sma:
            # df_graph = self.add_sma_to_def(df_graph, extra_mas=extra_mas)
            ma_sma7 = mpf.make_addplot(df_graph['sma7'], color='#3bed05', width=2.0)
            ma_sma77 = mpf.make_addplot(df_graph['sma77'], color='#ff7300', width=2.0)
            ma_sma200 = mpf.make_addplot(df_graph['sma200'], color='#001aff', width=2.0)
            ma_sma231 = mpf.make_addplot(df_graph['sma231'], color='#000000', width=2.0)
            if extra_mas:
                ma_sma100 = mpf.make_addplot(df_graph['sma100'], color="#8cf5ff", width=2.0)
                ma_sma279 = mpf.make_addplot(df_graph['sma279'], color='red', width=2.0)
                addplot = [ma_sma7, ma_sma77, ma_sma200, ma_sma231, ma_sma100, ma_sma279]
            else:
                addplot = [ma_sma7, ma_sma77, ma_sma200, ma_sma231]
        return df_graph, addplot

    def add_sma_to_df(self, df, extra_mas=False):
        self.strategy.load_data(df)

        df['sma7'] = self.strategy.sma7
        df['sma77'] = self.strategy.sma77
        df['sma200'] = self.strategy.sma200
        df['sma231'] = self.strategy.sma231
        if extra_mas:
            pass
            df['light blue'] = self.strategy.sma100
            df['sma279'] = self.strategy.sma279

        self.new = df[231:]

        return df[231:]

    def _plot_optimal_testing(self, profit, index=None, tf=True):
        if tf:
            print(profit.loc[index]['mean'].sort_values(ascending=False).head())
            profit.loc[index].plot(figsize=(20,6), xticks=range(35,350,10))
        else:
            print(profit['mean'].sort_values(ascending=False).head())
            profit.plot(figsize=(20,6), xticks=range(0,14,1))

    def plot_optimal_tf_testing(self, df, index):
        new = df.set_index(["delay", "symbol", "tf"])[['win loss rate', 'profitability', 'amount of data']] 
        new_df = new.sort_values(['delay', 'symbol', 'tf'])

        win_loss = new_df.reset_index().pivot(index=['delay', 'tf'], columns='symbol')['win loss rate']
        win_loss['mean'] = [win_loss.loc[index].median() for index in win_loss.index]
        profit = new_df.reset_index().pivot(index=['delay', 'tf'], columns='symbol')['profitability']

        profit['mean'] = [profit.loc[index].median() for index in profit.index]
        return self._plot_optimal_testing(profit, index, tf=True)

    def plot_optimal_delay_testing(self, df):
        df = df[df['profitability']>1]
        df = df[df['amount of data'] > 5]
        new = df.set_index(["delay", "symbol"])[['win loss rate', 'pnl/trade', 'amount of data']] 
        new_df = new.sort_values(['delay', 'symbol'])

        win_loss = new_df.reset_index().pivot(index='delay', columns='symbol')['win loss rate']
        win_loss['mean'] = [win_loss.loc[index].median() for index in win_loss.index]
        profit = new_df.reset_index().pivot(index='delay', columns='symbol')['pnl/trade']

        profit['mean'] = [profit.loc[index].median() for index in profit.index]
        self.testy = profit
        return self._plot_optimal_testing(profit, tf=False)

    def prepare_trade_graph_data(self, df_th, df_tf_candles, tid, tf, data_only=False, flat=False, adjust_left_view=29, adjust_right_view=10, extra_mas=False, save=False):
        start_datetime = df_th.loc[tid, 'enter_date']
        end_datetime = df_th.loc[tid, 'exit_date']
        enter_price = df_th.loc[tid, 'enter_price']
        exit_price = df_th.loc[tid, 'exit_price']
        position_side = df_th.loc[tid, 'side']
        date_column = df_tf_candles['date']
        base_tf = df_th['tf'].iloc[0]

        adjusted_pre_data_start_datetime = start_datetime - pd.Timedelta((231 * tf) + adjust_left_view * tf, 'minutes')
        adjusted_pre_data_end_datetime = start_datetime - pd.Timedelta(1 * tf, 'minutes')
        adjusted_post_data_start_datetime = end_datetime + pd.Timedelta(1 * tf, 'minutes')
        adjusted_post_data_end_datetime = end_datetime + pd.Timedelta(adjust_right_view * tf, 'minutes')

        pre_data = df_tf_candles[date_column.between(adjusted_pre_data_start_datetime, adjusted_pre_data_end_datetime)].copy().reset_index().set_index('date')
        post_data = df_tf_candles[date_column.between(adjusted_post_data_start_datetime, adjusted_post_data_end_datetime)].copy().reset_index().set_index('date')

        if df_tf_candles[date_column.between(start_datetime, end_datetime)]['date'].iloc[-1] < end_datetime:
            end_datetime += pd.Timedelta(1 * tf, 'minutes')
        trade_data = df_tf_candles[date_column.between(start_datetime, end_datetime)].copy().reset_index().set_index('date')

        trade_data['pnl_line'] = np.linspace(enter_price, exit_price, len(trade_data))
        illus_data = pre_data.append(trade_data.append(post_data))

        df_graph = self.add_sma_to_df(illus_data, extra_mas=extra_mas)

        if len(df_graph) == 0:
            return pd.DataFrame()

        if data_only:
            return df_graph

        self.graph_df(df_graph, position_side=position_side, flat=flat, tid=tid, save=save)
        return df_graph

    def graph_df(self, df_graph, position_side=None, no_sma=False, space=0, flat=False, tid=None, extra_mas=False, save=False):

        if df_graph.index.name != 'date':
            df_graph = df_graph.reset_index(drop=True).set_index('date')
        try:
            df_graph['sma7']
        except KeyError:
            df_graph = self.add_sma_to_df(df_graph, extra_mas=extra_mas)

        df_graph, addplot = self.set_graph_style(df_graph, sma=not no_sma)

        figratio = (40, 15)
        if flat:
            figratio = (40, 8)

        if position_side:
            pnl_line = df_graph['pnl_line'].dropna()

            if len(pnl_line) == 1:
                self.graph_df(df_graph, sma=True)
                return df_graph

            price_diff = pnl_line[-1]-pnl_line[-2]
            if (position_side == "Short" and price_diff<0) or (position_side == "Long" and price_diff>0):
                profit_line_inner = mpf.make_addplot(df_graph['pnl_line'], color='#00ff26', marker='o', markersize=50, width=4)
            else:
                profit_line_inner = mpf.make_addplot(df_graph['pnl_line'], color='red', marker='o', markersize=50, width=4)
            profit_line_outer = mpf.make_addplot(df_graph['pnl_line'], color='black', marker='o', markersize=50, width=7.0)

            addplot += [profit_line_outer, profit_line_inner]

        if save:
            symbol = df_graph.reset_index()['symbol'].iloc[0]
            file_name = f"C:\\Users\\haseab\\Desktop\\Python\\PycharmProjects\\FAB\\local\\PNG Images\\{symbol} #{tid}.png"

            graph = mpf.plot(df_graph, type='candle', figratio=figratio, datetime_format='%Y-%m-%d %H:%M:%S',
                             tight_layout=True, xrotation=0, xlim=(0, len(df_graph) + space), style=self.haseab_style,
                             warn_too_much_data=10000, addplot=addplot, savefig=file_name)


        graph = mpf.plot(df_graph, type='candle', figratio=figratio, datetime_format='%Y-%m-%d %H:%M:%S',
                         tight_layout=True, xrotation=180, xlim=(0, len(df_graph)+space), style=self.haseab_style,
                         warn_too_much_data=10000, addplot=addplot)
        return df_graph.reset_index()
