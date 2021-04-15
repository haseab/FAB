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

    def show_trade_graph(self, df_th, df_tf_candles, tid, save=False, adjust_left_view=29, adjust_right_view=10):
        start_datetime = df_th.loc[tid, 'enter_date']
        end_datetime = df_th.loc[tid, 'exit_date']
        enter_price = df_th.loc[tid, 'enter_price']
        exit_price = df_th.loc[tid, 'exit_price']
        position_side = df_th.loc[tid, 'side']
        tf = df_th['tf'].iloc[0]

        pre_data = df_tf_candles.loc[start_datetime - pd.Timedelta((231+adjust_left_view) * tf, 'minutes'): start_datetime -pd.Timedelta(1 * tf, 'minutes'), :].copy()
        post_data = df_tf_candles.loc[end_datetime + pd.Timedelta(1 * tf, 'minutes'): end_datetime + pd.Timedelta(adjust_right_view * tf, 'minutes'), :].copy()

        trade_data = df_tf_candles.loc[start_datetime:end_datetime, :].copy()
        trade_data['pnl_line'] = np.linspace(enter_price, exit_price, len(trade_data))
        illus_data = pre_data.append(trade_data.append(post_data))

        return self.graph_trade_data(illus_data, position_side, tid, save)


    def download_all_trades(self, df_trading_history, df_tf_candles, save=False):
        df_tf_candles = df_tf_candles.reset_index().set_index('date')
        df_th = df_trading_history.reset_index().set_index('tid')


        for tid in df_th.index:
            self.show_trade_graph(df_th, df_tf_candles, tid, save)
            print('saved file')
        return "Saved all of them"

    def graph_df(self, df_graph, sma=True):
        df_graph, addplot = self.set_graph_style(df_graph, sma=sma)
        graph = mpf.plot(df_graph, type='candle', figratio=(40, 15), datetime_format='%Y-%m-%d %H:%M:%S',
                         tight_layout=True, xrotation=5, style=self.haseab_style, addplot=addplot)

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
            addplot = [ma_green, ma_orange, ma_blue, ma_black]
        return df_graph, addplot

    def add_sma_to_def(self, df):
        self.strategy.load_data(df)
        self.strategy.update_moving_averages()

        df['green'] = self.strategy.green
        df['orange'] = self.strategy.orange
        df['blue'] = self.strategy.blue
        df['black'] = self.strategy.black
        return df[231:]

    def graph_trade_data(self, dataframe: pd.DataFrame, position_side, tid = None, save=False):
        """Graphs the selected data on a wide chart
        :return plot """
        df_graph = dataframe.copy()

        df_graph = self.add_sma_to_def(df_graph)

        if save:
            file_name = f"C:\\Users\\haseab\\Desktop\\Python\\PycharmProjects\\FAB\\local\\PNG Images\\BTCUSDT Trade No {tid}.png"

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

        df_graph, addplot = self.set_graph_style(df_graph, sma=True)

        if save:
            graph = mpf.plot(df_graph, type='candle', figratio=(40,15), datetime_format='%Y-%m-%d %H:%M:%S', tight_layout=True, xrotation =5,
                             style=self.haseab_style, addplot=addplot + [profit_line_outer, profit_line_inner], savefig=file_name)
        else:
            graph = mpf.plot(df_graph, type='candle', figratio=(40,15), datetime_format='%Y-%m-%d %H:%M:%S', tight_layout=True, xrotation =5,
                             style=self.haseab_style, addplot=addplot + [profit_line_outer, profit_line_inner])
        return df_graph
