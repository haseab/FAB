import matplotlib.pyplot as plt
import pandas as pd
from fab_strategy import FabStrategy
import mplfinance as mpf
from helper import Helper
import numpy as np
from datetime import timedelta

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

    def set_graph_style(self, df_graph, sma=True):
        addplot = []
        haseab_colors = mpf.make_marketcolors(up='#4ea672', down='#cf4532', edge='black')
        self.haseab_style = mpf.make_mpf_style(marketcolors=haseab_colors, gridcolor='#d4d4d4')

        if sma:
            # df_graph = self.add_sma_to_def(df_graph)
            ma_green = mpf.make_addplot(df_graph['green'], color='#3bed05', width=2.0)
            ma_orange = mpf.make_addplot(df_graph['orange'], color='#ff7300', width=2.0)
            ma_blue = mpf.make_addplot(df_graph['blue'], color='#001aff', width=2.0)
            ma_black = mpf.make_addplot(df_graph['black'], color='#000000', width=2.0)
            ma_light_blue = mpf.make_addplot(df_graph['light blue'], color="#8cf5ff", width=2.0)
            ma_red = mpf.make_addplot(df_graph['red'], color='red', width=2.0)

            addplot = [ma_green, ma_orange, ma_blue, ma_black, ma_light_blue, ma_red]
        return df_graph, addplot

    def add_sma_to_df(self, df):
        self.strategy.load_data(df)
        self.strategy.update_moving_averages()

        df['green'] = self.strategy.green
        df['orange'] = self.strategy.orange
        df['blue'] = self.strategy.blue
        df['black'] = self.strategy.black
        df['light blue'] = self.strategy.light_blue
        df['red'] = self.strategy.red

        self.new = df[231:]

        return df[231:]

    def prepare_trade_graph_data(self, df_th, df_tf_candles, tid, tf, data_only=False, flat=False, adjust_left_view=29, adjust_right_view=10, save=False):
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

        df_graph = self.add_sma_to_df(illus_data)

        if len(df_graph) == 0:
            return pd.DataFrame()

        if data_only:
            return df_graph

        self.graph_df(df_graph, position_side=position_side, flat=flat, tid=tid, save=save)
        return df_graph

    def graph_df(self, df_graph, position_side=None, sma=True, space=0, flat=False, tid=None, save=False):

        if df_graph.index.name != 'date':
            df_graph = df_graph.reset_index().set_index('date')
        try:
            df_graph['green']
        except KeyError:
            df_graph = self.add_sma_to_df(df_graph)

        df_graph, addplot = self.set_graph_style(df_graph, sma=sma)

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
