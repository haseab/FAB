# Roadmap

### Version 4.0 - MVP
1. [ ] Incorporate Database Functionality
2. [ ] Create Frontend Code
3. [ ] Don't hardcode the CSV fetching
4. [ ] Add rule where if rule 1 buy activates, check if price is below green, and don't buy until it goes higher
5. [ ] Compile all of this code into an executable
6. [ ] Add instance variables into constructor
7. [ ] Make set_asset consistent with Backtester and Trader class


### Version 4.1 - Backtesting Screener
1. [ ] Incorporate Unit Testing and TDD
2. [ ] Create Screener class
3. [ ] Backtest Screener: Collect all relevant statistics:
    - Volume
    - [Average,Minimum, Unrealized] Risk Reward Ratio
    - Average Profit / Average Loss
    - Largest Run / Largest Drawdown
    - Largest Profit / Largest Loss
    - Win Percentage / Total Trades
    - Win Streak / Losing Streak
    - Average Time for entire Trade
    - Average Time to Peak/Trough
    - Volume to Profit/Loss Ratio
    - Average Rate of Profit/Loss
    - Average Rate of Peak Profit/Peak Loss
4. [ ] Run Correlation algorithm to find most uncorrelated assets
5. [ ] Create Frontend portion of Backtesting Screener

### Version 4.2 - Live Screener
1. [ ] Create Live Screener Class (similar to TV)
2. [ ] Get relevant statistics for Live Screener 
    - Everything from Backtesting Screener Metrics
    - Volume
    - Net Change
    - Market Cap
    - Exchange 
    - Volatilty / Intraday Range
2. [ ] Create Frontend for Live Screener Class
3. [ ] Create Frontend OHLC Candles (Live View)
4. [ ] Create Functionality to show Backtested Trades on OHLC view


### Version 4.3 - Statistics and Machine Learning
1. [ ] Get educated on Quantitative Finance and all concepts & metrics
    - Kelly Criterion
    - Baum Welch Algorithm
    - Sharpe Ratios
    - Hidden Markov Models
2. [ ] Calculate confidence intervals of every single metric calculated
2. [ ] Partition Backtested Trades into finer details (not just by rules)
3. [ ] Calculate confidence intervals again for every metric
4. [ ] Automatic Stop Loss Generator
5. [ ] Find a way to predict consolidation
6. [ ] Create a bunch of regression ML models 
7. [ ] Implement multi objective decision making (maximizing profit in shortest amount of time)







### Ideas
1. [ ] Risk zones
2. [ ] Filtering daterange based on a common price movement
3. [ ] Voting system on timeframes
4. [ ] Dependent asset prices?
5. [ ] Sensitivity to seasonal or monthly or yearly changes?
6. [ ] incorporating unsupervised algorithm to gather patterns in price

