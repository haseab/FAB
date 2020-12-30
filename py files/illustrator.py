import matplotlib.pyplot as plt


class Illustrator():
    """
    Responsible for analysis of trading history.

    Attributes
    -----------
    None

    Methods
    ------------
    graph_data

    Please look at each method for descriptions
    """

    def graph_data(self, price: "dataframe"):
        """Graphs the selected data on a wide chart
        Returns: plot """
        plt.rcParams['figure.figsize'] = (40, 15)
        graph = plt.plot(price.index, price['Close'].values, '#848987', linewidth='0.75')
        return graph
