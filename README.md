# FAB - Financial Asset Bot

### What Problem is This Solving?
I started trading in 2016. Not only trading, I was **day** trading. Every trader knows the rollercoaster ride of emotions that comes with trading. You sell too early, you forget to put a stop, you forget that you even put a trade, you name it! We get emotional and just put up trades just for the sake of being in on the action. Well, after tens of thousands of dollars lost due to a lack of strategy and discipline when trading, I learned that it's best to not personally trade ever again. 

But then I had an idea. Yes **I will personally** never trade ever again. But why can't a bot that I created trade for me? In fact programming a bot to trade will automatically solve the biggest two issues as to why I was losing money: No adherence to a strategy, and being emotional.  

### My Vision with This Bot
I personally see the development of this bot not only as a passionate past time, but also as an investment in my future. I realize that if I can optimize for growth rate over income, that I can escape the rat race and do what I love. I can have a robot trade for me while I sleep, as well as while I go enjoy the sweetness of life. 

This bot not only comes with the trading capability, but with backtesting capability which is one of the most powerful implementations. It can test any strategy over the history of any asset, in a matter of seconds. There are many things I have yet to add, such as diversification of assets (reducing risk considerably while maintaining the same reward!), machine learning and statistical analyses, as well as an asset screener that optimally shows the most promising trades according to a range of metrics. As you're reading this, I'm educating myself on quantitative finance so that eventually, I'll able to take this bot to the level that quants on wall street can.

If you have an aligned vision of optimizing for growth rate, or are interesting in helping develop this, feel free to fork this repo. Also don't hesitate to reach out to me.

## Table of Contents
[Overview](#Overview)
- [Structure](#Structure)
- [Features](#Features)
- [Requirements](#Requirements)

[Example](#Example)
 - [Backtesting](#Backtesting)
 - [Trading](#Trading)


## Overview
### Structure
Below is a UML Class Diagram that gives a high level understanding of how this system works together. 

![](https://github.com/haseab/FAB/blob/master/example%20images/uml_class_diagram.png)

### Features
This program has the following features:

1. **Trading Strategy** 
    - A trading strategy is essentially a set of conditions that a computer looks for when choosing to enter or exit a position. 
    - Ex. Buy when Price goes 2% up in 5 minutes. Close when Price goes down 0.5% in 1 minute. 
    - Ex. Buy when 10 MA crosses above the 50 MA. Close when 10 MA crosses below the 50 MA.
    - These strategies must be so specific that a computer can read it and execute what you want. 
    - The strategy is an object in a separate file and all you have to do is to call the strategy into the "Trader" class (which will be introduced shortly) and it will automatically start trading in reference to it. The only conditions are that it must boolean values for its methods. 
  
2. **DataLoader** 
    - This fundamental feature is the all inclusive ETL solution to extracting the necessary data to input into either the Backtester or the Trader class.
    - The DataLoader requires the historical data in a 1 min OHLC candlestick format. This could be either in a CSV or straight from a request to an exchange API.
    - The DataLoader also has methods for cleaning data, abstracting data, etc. 

3. **Backtester** 
    - The Backtester applies the trading strategy onto testing data as a measure to see what trades would have been executed in the past, assuming the strategy ran then. This is  invaluable as it provides a non bias approach to how the strategy would have executed since the inputs types are constant.
    - Note: One mistake to avoid when backtesting is to create your strategy with the same data that you'll be backtesting it on. This can cause an overfitting bias. 

3. **Trader** 
    - The Trader is the most integral part of this system. It's what brings the excitement of being able to trade in one's sleep. This complex class contains methods that are responsible for executing on the buy/sell/short/cover decisions that are made from the Trading Strategy. 
 
4. **Analyzer** - Summarizing results of the Backtester or Trader
    - After all trades are made from either the Backtester or Trader, the trading history needs to be analyzed somehow. Metrics including but not limited to risk reward ratio, average drawdown, win loss ratio, and most importantly, the profit margin are what is calculated in this class.

 
### Requirements
- Attached is the [requirements.txt file](#https://github.com/haseab/FAB/blob/master/requirements.txt)
- This code was made with Python 3.8
- The following non-native modules were used:
  - [matplotlib](#https://pypi.org/project/matplotlib/)
  - [pandas](#https://pypi.org/project/pandas/)
  - [numpy](#https://pypi.org/project/numpy/)
  - [python-binance](#https://pypi.org/project/python-binance/)
  

An example trading strategy will be shown to illustrate how both of these functionalities are executed. This strategy is my own personal algorithm that I've tweaked over the years and it is very profitable! 
Might sound crazy, but this strategy made about 110% profit in the year that Bitcoin had its bear market.  However this code should mainly serve as a wrapper for your own trading strategy.

## Example
The example.py file is a file that illustrates the backtesting, as well as the trading features that this trading bot offers.

### Importing Libraries 
In order to trade or even backtest, you need only the strategy class and the backtest/trader classes. The rest of the classes are already imported in the other files.
<pre>
from fab_strategy import FabStrategy
from backtester import Backtester
from trader import Trader
</pre>

### Backtesting
A number of things must be done in order to backtest successfully. You need to 
- set the asset (e.g. BTCUSDT, ETHUSDT, LTCUSDT)
- set the date range (the time period in which you want to test)
- set the timeframe (how big of a period you want the candles to represent)

<pre>
b = Backtester()
b.set_asset("BTCUSDT")
b.set_timeframe(77)
b.set_date_range("2018-01-01", "2019-01-01")
</pre>

**Note**: It is important to run these commands in order. 
- You can't set the range without setting the timeframe first. 
- You can't set the timeframe without setting the asset first


Once those are done, you can initiate backtesting by the following command: 
 <pre>
strategy = FabStrategy()
sensitivity = 0.0001

results = b.start_backtest(strategy, sensitivity)
print(results)
</pre>

This will start the backtesting by going through the entire dataset and see if your rules from the <code>FabStrategy</code> apply. It will then calculate a load of important metrics than can be accessed after.

The final return value should look something like this:

![](https://github.com/haseab/FAB/blob/master/example%20images/trade_stats_example.png)


#### Further Additions

You can get further metrics after running this backtest. Among the many things calculated, you can obtain the trading history of exactly what was bought at what time. Below is an image displaying this:

![](https://github.com/haseab/FAB/blob/master/example%20images/trade_list_example.png)

You can also directly access all the metrics that were listed in the summary. Just take a look at the documentation for their attributes:

Example: 
<pre>
>>> print(b.profit)
1.997
</pre>

### Live Trading

Live trading is actually a little simpler than backtesting. There is no information to load, that is already done for you. There is no need to set date ranges since it's live. 
The only thing that is needed to do is to set the asset, timeframe and leverage. Leverage by default is going to be 0.001x what you have in your account for safety reasons. 

<pre>
t = Trader() 
t.load_account() 
t.set_timeframe(77)  
t.set_asset("BTCUSDT")</pre> 

**Note**: You must also have a configuration file with your Binance API keys. Below is an example of how the txt file should look like
<pre>
API_KEY,"FASDJFLASDJLFKAJ;DSKFJ;ASDLKFJ;ALDSKF"
SECRET,"SD;FJSDALKFJ;ASLDKJFLSDKJFLASKDJFOLAS"
</pre>


**Note**: the timeframe needs to be set before the setting of the asset. This is because it will fetch this information from the API rather than using 1 minute data.

When you're ready to trade, you run:
<pre>
strategy = FabStrategy()
t.start_trading(strategy)
</pre>

And it will automatically trade according to the strategy that is inputted. 

That is it so far, enjoy! 