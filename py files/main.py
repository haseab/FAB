from dataload import DataLoader
from backtester import Backtester

csvUrl = "INSERT URL"

data = DataLoader()
print(data.load_csv(csvUrl))
