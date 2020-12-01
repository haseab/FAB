from dataloader import DataLoader
from illustrator import Illustrator
from strategy_executor import StrategyExecutor
from analyzer import Analyzer

# Instantiating
load1 = DataLoader()
analyze1 = Analyzer()
illustrate1 = Illustrator()
executor1 = StrategyExecutor()

# Setting vars
csvUrl = "bitcoin_data_2018_1_1_to_2018_5_1.csv"
data = load1.load_csv(csvUrl)
start = "2018-01-01"
end = "2018-05-01"
tf = 77
dfraw = load1.get_range(data, start, end)
df = load1.timeframe_setter(dfraw, tf)

# Graphing Data
illustrate1.graph_data(dfraw)

# Getting Trade History
trade_history = executor1.fab_strategy(df)

# Getting Statistics
analyze1.calculate_statistics(trade_history)
print(analyze1.summarize_statistics())

