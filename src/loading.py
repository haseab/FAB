import concurrent.futures as conc
from binance.client import Client
import time


def threading_test(func1, func2, *args):
    with conc.ThreadPoolExecutor(max_workers=5) as executor:
        f1 = executor.submit(func1, *args)
        f2 = executor.submit(func2, *args)
    return f1,f2

if __name__ == "__main__":
    client1 = Client()
    client2 = Client()
    symbol = 'BTCUSDT'
    tf = '1m'

    start = time.time()
    f1, f2 = threading_test(client1.get_historical_klines, client2.get_historical_klines, symbol, tf, "5000 minutes ago")
    end = time.time()
    print(end-start)
    print(f1.result()[:10])
    print(f2.result()[:10])

