import numpy as np


class Coppock(object):
    def __init__(self, wma_pd=10, roc_long=14, roc_short=11, hist='historical_data/hist_data.npy'):
        self.wma_pd = wma_pd
        self.long_pd = roc_long
        self.short_pd = roc_short
        self.past_data = None

        self.values = self.calc_copp(np.load(hist))

    def calc_copp(self, data):
        short_roc = [0] * (self.short_pd - 1)
        long_roc = [0] * (self.long_pd - 1)

        # obtain shorter roc change values
        for idx in range(self.short_pd, len(data)):
            short_roc.append((data[idx] - data[idx - self.short_pd]) / data[idx - self.short_pd] * 100)

        # obtain longer roc change values
        for idx in range(self.long_pd, len(data)):
            long_roc.append((data[idx] - data[idx - self.long_pd]) / data[idx - self.long_pd] * 100)

        # caculate WMA of roc change sum
        copp = [0] * (self.long_pd + self.wma_pd)
        roc_sum = np.add(short_roc, long_roc)

        for idx in range(self.long_pd + self.wma_pd, len(roc_sum) + 1):

            # EMA based on set period
            temp_wma = 0
            weight_sum = 0
            for i in range(self.wma_pd):
                temp_wma += (roc_sum[(idx - self.wma_pd) + i] * (i + 1))
                weight_sum += i + 1

            copp.append(temp_wma / weight_sum)

        # save data for update calculation
        self.past_data = data[-(self.long_pd + self.wma_pd):]

        return np.array(copp)

    def update_copp(self, value):
        # update data for calculations
        self.past_data.append(value)
        self.past_data.pop(0)

        short_roc = [0] * self.short_pd
        long_roc = [0] * self.long_pd

        # obtain shorter roc change values
        for idx in range(self.short_pd, len(self.past_data)):
            short_roc.append((self.past_data[idx] - self.past_data[idx - self.short_pd]) / self.past_data[idx - self.short_pd] * 100)

        # obtain longer roc change values
        for idx in range(self.long_pd, len(self.past_data)):
            long_roc.append((self.past_data[idx] - self.past_data[idx - self.long_pd]) / self.past_data[idx - self.long_pd] * 100)

        roc_sum = np.add(short_roc, long_roc)

        # EMA based on set period
        temp_wma = 0
        weight_sum = 0
        index = len(roc_sum) - self.wma_pd
        for i in range(self.wma_pd):
            temp_wma += (roc_sum[index + i] * (i + 1))
            weight_sum += i + 1

        return temp_wma / weight_sum
