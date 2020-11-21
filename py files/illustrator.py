class Illustrator():
    #     def __init__(self, data):
    #         self.data - data

    def graph_data(self, price):
        """Graphs the selected data on a wide chart
        Returns: plot """
        plt.rcParams['figure.figsize'] = (40, 15)
        graph = plt.plot(price.index, price.values, '#848987', linewidth='0.75')
        return graph
