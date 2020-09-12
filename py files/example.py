from dataload import DataLoad
from backtester import Backtester
from bm_strategy import *


csvUrl = "bitcoin_data_2018_1_1_to_2018_5_1.csv"

load1 = DataLoad()
data = load1.load_csv(csvUrl)
strategy = bm_strategy(load1, data,7,77,231,880,2354,77,"2018-01-01","2018-05-01")

backtest_bm = Backtester(data,strategy)


print(backtest_bm.trade_stats(strategy))
