import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import copy


class Rsi(object):
    def __init__(self, data, period=14):
        self.period = period
        self.prev_losses = None
        self.prev_gains = None
        self.last_price = None

        self.value = self.calc_rsi(data)

    def get_rs(self, data):
        # find the gain and loss on each point
        losses, gains = [0], [0]
        for idx in range(1, len(data)):
            change = data[idx] - data[idx - 1]

            # check if change is loss or gain
            if change >= 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        return losses, gains

    def calc_rs(self):
        # calculate averages to get relative strength
        avg_loss = np.average(self.prev_losses)
        avg_gain = np.average(self.prev_gains)

        if avg_loss == 0:
            return 100

        return avg_gain / avg_loss

    def calc_rsi(self, data):
        # initialize for incomplete range
        rsi = [0] * (self.period - 1)

        # get gains and losses on data
        losses, gains = self.get_rs(data)

        for idx in range(self.period, len(data) + 1):
            # save for update function
            self.prev_losses = losses[idx - self.period:idx]
            self.prev_gains = gains[idx - self.period:idx]
            self.last_price = data[idx - 1]

            rs = self.calc_rs()
            rsi.append(100 - (100 / (1 + rs)))

        return rsi

    def update_rsi(self, price):
        change = price - self.last_price
        self.last_price = price

        if change >= 0:
            self.prev_losses.append(0)
            self.prev_gains.append(change)
        else:
            self.prev_losses.append(abs(change))
            self.prev_gains.append(0)

        # remove oldest value
        self.prev_gains.pop(0)
        self.prev_losses.pop(0)

        rs = self.calc_rs()
        self.value.append(100 - (100 / (1 + rs)))
        self.value.pop(0)

        return self.value[-1]

class StochRsi(Rsi):
    def __init__(self, data, period=14):
        super(StochRsi, self).__init__(data, period=period)

        self.ma_k = None
        self.ma_d = None

        self.smooth_k = 3
        self.smooth_d = 3

        self.stoch_value = self.calc_stoch_rsi()
        self.hist_values = self.calc_histo()

    def calc_stoch_rsi(self):
        stoch_rsi = [0] *  (self.period*2 - 2)

        # find StochRSI based on period low and high
        for idx in range(self.period*2 - 1, len(self.value) + 1):
            window = self.value[idx-self.period:idx]
            high = np.amax(window); low = np.amin(window)
            stoch_rsi.append((self.value[idx-1] - low)/ (high - low))

        return stoch_rsi

    def calc_histo(self):
        # find K%
        self.ma_k = [0] * (self.smooth_k - 1)
        for idx in range(self.smooth_k, len(self.stoch_value) + 1):
            self.ma_k.append(np.average(self.stoch_value[idx - self.smooth_k:idx]))

        # find D%
        self.ma_d = [0] * (self.smooth_d - 1)
        for idx in range(self.smooth_d, len(self.ma_k) + 1):
            self.ma_d.append(np.average(self.ma_k[idx - self.smooth_d:idx]))

        # subtract two arrays
        return [x1 - x2 for (x1, x2) in zip(self.ma_k, self.ma_d)]

    def update_stoch_rsi(self, price):
        _ = super(StochRsi, self).update_rsi(price)

        window = self.value[-self.period:]
        high = np.amax(window); low = np.amin(window)

        self.stoch_value.append((self.value[-1] - low) / (high - low))
        self.stoch_value.pop(0)

    def update_stoch_hist(self, price):
        self.update_stoch_rsi(price)

        self.ma_k.append(np.average(self.stoch_value[-self.smooth_k:]))
        self.ma_d.append(np.average(self.ma_k[-self.smooth_d:]))

        self.ma_k.pop(0)
        self.ma_d.pop(0)

        return self.ma_k[-1] - self.ma_d[-1]
