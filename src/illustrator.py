import matplotlib.pyplot as plt
import pandas as pd
from fab_strategy import FabStrategy
import mplfinance as mpf
from helper import Helper
import numpy as np


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

    def show_trade_graph(self, df_th, df_tf_candles, tid, save=False, large_view=False):
        start_datetime = df_th.loc[tid, 'enter_date']
        end_datetime = df_th.loc[tid, 'exit_date']
        enter_price = df_th.loc[tid, 'enter_price']
        exit_price = df_th.loc[tid, 'exit_price']
        position_side = df_th.loc[tid, 'side']
        tf = df_th['tf'].iloc[0]

        if large_view:
            pre_data = df_tf_candles.loc[start_datetime - pd.Timedelta(400 * tf, 'minutes'): start_datetime -pd.Timedelta(1 * tf, 'minutes'), :].copy()
            post_data = df_tf_candles.loc[end_datetime + pd.Timedelta(1 * tf, 'minutes'): end_datetime + pd.Timedelta(100 * tf, 'minutes'), :].copy()
        else:
            pre_data = df_tf_candles.loc[start_datetime - pd.Timedelta(260 * tf, 'minutes'): start_datetime -pd.Timedelta(1 * tf, 'minutes'), :].copy()
            post_data = df_tf_candles.loc[end_datetime + pd.Timedelta(1 * tf, 'minutes'): end_datetime + pd.Timedelta(10 * tf, 'minutes'), :].copy()

        trade_data = df_tf_candles.loc[start_datetime:end_datetime, :].copy()
        trade_data['pnl_line'] = np.linspace(enter_price, exit_price, len(trade_data))

        illus_data = pre_data.append(trade_data.append(post_data))

        self.graph_trade_data(illus_data, position_side, tid, save)


    def download_all_trades(self, df_trading_history, df_tf_candles, save=False):
        df_tf_candles = df_tf_candles.reset_index().set_index('date')
        df_th = df_trading_history.reset_index().set_index('trade_id')


        for tid in df_th.index:
            self.show_trade_graph(df_th, df_tf_candles, tid, save)
            print('saved file')
        return "Saved all of them"

    def graph_df(self, df_graph):
        haseab_colors = mpf.make_marketcolors(up='#4ea672', down='#cf4532', edge='black')
        haseab_style = mpf.make_mpf_style(marketcolors=haseab_colors, gridcolor='#d4d4d4')

        graph = mpf.plot(df_graph, type='candle', figratio=(40, 15), datetime_format='%Y-%m-%d %H:%M:%S', tight_layout=True, xrotation=5, style=haseab_style)

    def graph_trade_data(self, dataframe: pd.DataFrame, position_side, tid = None, save=False):
        """Graphs the selected data on a wide chart
        :return plot """

        df_graph = dataframe.copy()

        self.strategy.load_data(df_graph)
        self.strategy.update_moving_averages()

        df_graph['green'] = self.strategy.green
        df_graph['orange'] = self.strategy.orange
        df_graph['blue'] = self.strategy.blue
        df_graph['black'] = self.strategy.black

        df_graph = df_graph[231:]

        if save:
            file_name = f"C:\\Users\\haseab\\Desktop\\Python\\PycharmProjects\\FAB\\local\\PNG Images\\BTCUSDT Trade No {tid}.png"

        pnl_line = df_graph['pnl_line'].dropna()
        price_diff = pnl_line[-1]-pnl_line[-2]

        if (position_side == "Short" and price_diff<0) or (position_side == "Long" and price_diff>0):
            profit_line_inner = mpf.make_addplot(df_graph['pnl_line'], color='#00ff26', marker='o', markersize=50, width=4)
        else:
            profit_line_inner = mpf.make_addplot(df_graph['pnl_line'], color='red', marker='o', markersize=50, width=4)

        # up = '#6BA583', down = '#D75442' (Tradingview Colours)
        haseab_colors = mpf.make_marketcolors(up='#4ea672', down='#cf4532', edge='black')
        haseab_style = mpf.make_mpf_style(marketcolors=haseab_colors, gridcolor='#d4d4d4')
        ma_green = mpf.make_addplot(df_graph['green'], color='#3bed05', width=2.0)
        ma_orange = mpf.make_addplot(df_graph['orange'], color='#ff7300', width=2.0)
        ma_blue = mpf.make_addplot(df_graph['blue'], color='#001aff', width=2.0)
        ma_black = mpf.make_addplot(df_graph['black'], color='#000000', width=2.0)
        profit_line_outer = mpf.make_addplot(df_graph['pnl_line'], color='black', marker='o', markersize=50, width=7.0)

        if save:
            graph = mpf.plot(df_graph, type='candle', figratio=(40,15), datetime_format='%Y-%m-%d %H:%M:%S', tight_layout=True, xrotation =5,
                             style=haseab_style, addplot=[ma_green, ma_orange, ma_blue, ma_black,profit_line_outer, profit_line_inner], savefig=file_name)
        else:
            graph = mpf.plot(df_graph, type='candle', figratio=(40,15), datetime_format='%Y-%m-%d %H:%M:%S', tight_layout=True, xrotation =5,
                             style=haseab_style, addplot=[ma_green, ma_orange, ma_blue, ma_black,profit_line_outer, profit_line_inner])
        return graph