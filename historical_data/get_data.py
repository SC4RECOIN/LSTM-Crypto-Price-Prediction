from binance.client import Client
import numpy as np


# load key and secret and connect to API
keys = open('../keys.txt').readline()
api = Client(keys[0], keys[1])

# fetch 15 minute candles of all data
hist = api.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE, "17 Aug, 2017")

# create numpy object with closing prices and volume
hist = np.array(hist)
vol = hist[:, 5]
hist = hist[:, 4]

# data information
print("\nDatapoints:  {0}".format(hist.shape[0]))
print("Memory:      {0:.2f} Mb\n".format(hist.nbytes / 1000000))

# save to file as numpy object
np.save("hist_data", hist)
np.save("hist_volume", vol)
