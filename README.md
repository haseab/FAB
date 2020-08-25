# FAB - Financial Asset Bot

## What is this Code About?
This program has two functions
### 1. Backtesting Functionality - Testing algorithms with historical data 
  - The Backtesting Functionality requires a CSV of historical 1 min data of any financial asset. It can then apply the trading algorithm onto that testing data as a measure to see what trades would have been executed in the past, given that the algorithm was running. 
    
### 2. Trading Functionality - Executing trades on the basis of those algorithms
  This program can connect to an exchange using an API key, and make REST API calls to grab information or to make trades based off of the algorithms decisions
  

An example algorithm will be shown to illustrate how both of these functionalities are executed. This original algorithm is very profitable! Making about 110% profit in one year in the Bitcoin markets from Apr 2019 to Apr. 2020. However this code should mainly serve as a wrapper for your own trading strategy.
