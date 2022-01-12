from backtester import Backtester
from screener import Screener


class Screentester:

  def __init__():
    pass

  def optimize_trades(self, current_positions, leverage, df_metrics, trades_to_enter, number_of_trades):
    if len(current_positions) != 0:
        self.capital = current_positions['usd size'].sum()
        trades_to_close, final_trades_to_enter = self._profit_optimization(df_metrics, current_positions, trades_to_enter, number_of_trades)
        if trades_to_close:
            self.close_trade_info = executor.exit_market(self.binance, self.get_positions_amount(trades_to_close))
            display(self.close_trade_info)
        if trades_to_enter:
            dividing_factor = (len(current_positions) + len(trades_to_enter))/len(current_positions)
            positions_amount = self.get_positions_amount(current_positions['symbol'].values, divide_by=dividing_factor)
            print(positions_amount)

            self.exit_trade_info = self.exit_trade_info.append(executor.exit_market(self.binance, positions_amount))
            self.enter_trade_info = self.enter_trade_info.append(executor.enter_market(self.binance, final_trades_to_enter, self.capital, leverage, number_of_trades))
            display(self.enter_trade_info)
        else:
            print("No Trades made")

  def run_backtest(symbols, ):
    
    for symbol in symbols:
      pass
