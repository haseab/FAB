# FAB - Financial Asset Bot

### What Problem is This Solving?
I started trading in 2016. Not only trading, I was **day** trading. Every trader knows the rollercoaster ride of emotions that comes with trading. You sell too early, you forget to put a stop, you forget that you even put a trade, you name it! We get emotional and just put up trades just for the sake of being in on the action. Well, after tens of thousands of dollars lost due to a lack of strategy and discipline when trading, I learned that it's best to not personally trade ever again. 

But then I had an idea. Yes **I will personally** never trade ever again. But why can't a bot that I created trade for me? In fact programming a bot to trade will automatically solve the biggest two issues as to why I was losing money: No adherence to a strategy, and being emotional.  

This bot not only comes with the trading capability, but with backtesting capability which is one of the most powerful implementations. It can test any strategy over the history of any asset, using algorithms. The code here is modular enough for machine learning and AI algorithms to be plugged in as well and backtested. 

Ever since I have automated my trading, I have been consistently making profit. I made **100%** in the crypto markets last year alone. 

## Table of Contents
- [Overview](#Overview)
  - [Description](#Description)
  - [Requirements](#Requirements)

- [Examples](#Examples)
  - [Importing Modules](#Importing-Modules)
  - [Loading Data](#Loading-Data)
  - [Loading Strategy](#Loading-Strategy)
  - [Backtesting Execution](#Backtesting-Execution)
  - Connnecting to Exchange (coming soon)
  - Automatically putting Buy/Sell orders (coming soon)

## Overview
### Description
This program has two functions

1. **Backtesting Functionality** - Testing algorithms with historical data 
  - The Backtesting Functionality requires a CSV of historical 1 min data of any financial asset. It can then apply the trading algorithm onto that testing data as a measure to see what trades would have been executed in the past, given that the algorithm was running. 
    
2. **Trading Functionality** - Executing trades on the basis of those algorithms
  - This program can connect to an exchange using an API key, and make REST API calls to grab information or to make trades based off of the algorithms decisions
 
### Requirements
- This code was made with Python 3.8
- The following non-native modules were used:
  - [matplotlib](#https://pypi.org/project/matplotlib/)
  - [pandas](#https://pypi.org/project/pandas/)
  - [numpy](#https://pypi.org/project/numpy/)

An example algorithm will be shown to illustrate how both of these functionalities are executed. This original algorithm is very profitable! Making about 110% profit in one year in the Bitcoin markets from Apr 2019 to Apr. 2020. However this code should mainly serve as a wrapper for your own trading strategy.

## Examples
please check 'example.py' for a similar example.

### Importing Modules
Each module is a different class (except for the last one, which is a function). 
- **DataLoad function:** to get the range, timeframe and chart of any dataset that you need
- **bm_strategy:** My personal trading method added as an example to execute the backtester
- **Backtester function:** to use the strategy on historical data to see historical performance

Importing the python files in this project
<pre>
from dataload import DataLoad
from backtester import Backtester
from bm_strategy import *
</pre>

### Loading Data 
Loading data here requires getting csvUrl, creating an instance of the DataLoad class and then loading the csv
<pre>
csvUrl = "bitcoin_data_2018_1_1_to_2018_5_1.csv"
load1 = DataLoad()
data = load1.load_csv(csvUrl)
</pre>

If the data ever needs to be visualized, that can also be done by the following command

<pre> data.graph_data()</pre>

![](https://github.com/haseab/FAB/blob/master/example%20images/chart_example.png)


### Loading Strategy
Loading the strategy is the hardest part here. Whatever strategy that is inputted must follow a specific format
A picture below is as follows

![](https://github.com/haseab/FAB/blob/master/example%20images/trade_list_example.png)

<pre>
strategy = bm_strategy(load1, data,7,77,231,880,2354,77,"2018-01-01","2018-05-01")
</pre>

### Backtesting Execution
Create an instance of the Backtester class, then find the trade statistics

<pre>
backtest_bm = Backtester(data,strategy)
print(backtest_bm.trade_stats(strategy))
</pre>

and the result should be as follows

![](https://github.com/haseab/FAB/blob/master/example%20images/trade_stats_example.png)
