import numpy as np


class Ema(object):
    def __init__(self, period):
        self.period = period
        self.k = 2 / (period + 1)
        self.prev_ema = None

    def calc_ema(self, data):
        # initialize seed and first value
        seed = sum(data[:self.period]) / self.period
        ema = [(data[0] * self.k) + (seed * (1 - self.k))]

        # find next ema based on previous ema and weight
        for price in data[1:]:
            ema.append((price * self.k) + (ema[-1] * (1 - self.k)))

        # save the last ema for update function
        self.prev_ema = ema[-1]

        return np.array(ema)

    def update_ema(self, value):
        # update ema
        self.prev_ema = (value * self.k) + (self.prev_ema * (1 - self.k))

        return self.prev_ema

class Macd(object):
    def __init__(self, data, short_pd=12, long_pd=26, sig_pd=9):
        self.long_ema = Ema(long_pd)
        self.short_ema = Ema(short_pd)
        self.signal_ema = Ema(sig_pd)

        # find macd histogram
        self.values = self.calc_macd(data)

    def calc_macd(self, data):
        # calculate short and long ema
        short_ema = self.short_ema.calc_ema(data)
        long_ema = self.long_ema.calc_ema(data)

        # calculate macd and signal
        macd = short_ema - long_ema
        signal = self.signal_ema.calc_ema(macd)

        # return histogram
        return macd - signal

    def update_macd(self, value):
        # update ema
        self.long_ema.update_ema(value)
        self.short_ema.update_ema(value)

        # find macd and signal
        macd = self.short_ema.prev_ema - self.long_ema.prev_ema
        self.signal_ema.update_ema(macd)

        # return histogram
        return macd - self.signal_ema.prev_ema

class PPO(object):
    def __init__(self, data, short_pd=12, long_pd=26, sig_pd=9):
        self.long_ema = Ema(long_pd)
        self.short_ema = Ema(short_pd)
        self.signal_ema = Ema(sig_pd)

        # find histogram
        self.values = self.calc_ppo(data)

    def calc_ppo(self, data):
        # calculate short and long ema
        short_ema = self.short_ema.calc_ema(data)
        long_ema = self.long_ema.calc_ema(data)

        # calculate ppo and signal
        ppo = ((short_ema - long_ema) / long_ema) * 100
        signal = self.signal_ema.calc_ema(ppo)

        # return histogram
        return ppo - signal

    def update_ppo(self, value):
        # update ema
        self.long_ema.update_ema(value)
        self.short_ema.update_ema(value)

        ppo = ((self.short_ema.prev_ema - self.long_ema.prev_ema) / self.long_ema.prev_ema) * 100
        self.signal_ema.update_ema(ppo)

        # return histogram
        return ppo - self.signal_ema.prev_ema


if __name__ == "__main__":
    data = np.load('data/hist_data.npy')[:, 4]
    test1 = PPO(data, 6, 12, 3).values

    update_data = data[-20:]
    test2 = PPO(data[:-20], 6, 12, 3)
    test2_values = test2.values

    updated = [x for x in test2_values]

    for value in update_data:
        updated.append(test2.update_ppo(value))

    import matplotlib.pyplot as plt
    plt.plot(test1[-100:])
    plt.plot(updated[-100:])
    plt.show()