import numpy as np
import plotly.offline as py
import plotly.graph_objs as go


class Rsi(object):
    def __init__(self, period=14, hist='historical_data/hist_data.npy'):
        self.period = period
        self.prev_losses = None
        self.prev_gains = None
        self.last_price = None

        self.value = self.calc_rsi(np.load(hist))

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

        return np.array(rsi)

    def update_rsi(self, value):
        change = value - self.last_price
        self.last_price = value

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

        return 100 - (100 / (1 + rs))

class StochRsi(Rsi):
    def __init__(self, period=14):
        super(StochRsi, self).__init__(period=period)

        self.stoch_value = self.calc_stoch_rsi()
        self.values = self.calc_histo()
    
    def calc_stoch_rsi(self):
        stoch_rsi = [0] *  (self.period*2 - 2)

        # find StochRSI based on period low and high
        for idx in range(self.period*2 - 1, len(self.value) + 1):
            window = self.value[idx-self.period:idx]
            high = np.amax(window); low = np.amin(window)
            stoch_rsi.append((self.value[idx-1] - low)/ (high - low))

        return np.array(stoch_rsi)
    
    def calc_histo(self, pd=3):
        # find %K
        ma_k = np.copy(self.stoch_value)
        for idx in range(3,len(ma_k) + 1):
            ma_k[idx-1] = np.average(ma_k[idx - 3:idx])
        
        # find %D
        ma_d = np.copy(ma_k)
        for idx in range(3,len(ma_d) + 1):
            ma_d[idx-1] = np.average(ma_d[idx - 3:idx])
        
        return ma_k - ma_d
