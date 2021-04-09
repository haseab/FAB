import matplotlib.pyplot as plt
import pandas as pd
from fab_strategy import FabStrategy
import mplfinance as mpf

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


    def graph_data(self, dataframe: pd.DataFrame):
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

        haseab_colors = mpf.make_marketcolors(up='g', down='r', edge='black')
        haseab_style = mpf.make_mpf_style(marketcolors=haseab_colors, gridcolor='#d4d4d4')
        ma_green = mpf.make_addplot(df_graph['green'], color='#3bed05', width=2.0)
        ma_orange = mpf.make_addplot(df_graph['orange'], color='#ff7300', width=2.0)
        ma_blue = mpf.make_addplot(df_graph['blue'], color='#001aff', width=2.0)
        ma_black = mpf.make_addplot(df_graph['black'], color='#000000', width=2.0)

        graph = mpf.plot(df_graph, type='candle', figratio=(40,15), style=haseab_style, addplot=[ma_green, ma_orange, ma_blue, ma_black])
        return graph