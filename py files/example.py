from dataload import DataLoad
from backtester import Backtester
from bm_strategy import *

# Instantiating
load1 = DataLoader()
analyze1 = Analyzer()
illustrate1 = Illustrator()
executor1 = StrategyExecutor()

# Setting vars
data = load1.load_csv(csvUrl)
csvUrl = "bitcoin_data_2018_1_1_to_2018_5_1.csv"
start = "2018-01-01"
end = "2018-05-01"
tf = 77
backtest_bm = Backtester(data,strategy)
dfraw = load1.get_range(data, start, end)
df = load1.timeframe_setter(dfraw, tf)

# Graphing Data
illustrate1.graph_data(dfraw)

# Getting Trade History
trade_history = backtest.fab_strategy(df)

# Getting Statistics
analyze1.calculate_statistics(trade_history)
print(analyze1.summarize_statistics())

