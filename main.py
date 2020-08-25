from dataload import DataLoad
from backtester import Backtester

csvUrl = "INSERT URL"

data = DataLoad()
print(data.load_csv(csvUrl))
