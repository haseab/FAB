import matplotlib.pyplot as plt
import pandas as pd


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

    def graph_data(self, price: pd.DataFrame):
        """Graphs the selected data on a wide chart
        :return plot """
        plt.rcParams['figure.figsize'] = (40, 15)
        graph = plt.plot(price.index, price['Close'].values, '#848987', linewidth='0.75')
        return graph
