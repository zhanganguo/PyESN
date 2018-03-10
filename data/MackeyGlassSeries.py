from data.Data import Data
import numpy as np
from pylab import *


class MackeyGlassSeries(Data):
    def __init__(self):
        Data.__init__(self)

    def generate_data(self, tau, series_length):
        init_washout_length = 1000
        incremental_per_unit = 10
        gen_history_length = tau * incremental_per_unit
        x1 = 1.2 * np.ones((gen_history_length, 1))
        x2 = 0.2 * np.subtract(np.random.rand(gen_history_length, 1), 0.5)
        seed = np.add(x1, x2)
        old_value = 1.2
        gen_history = seed
        speed_up = 1

        mgs = np.zeros((series_length, 1))

        step = 0

        for n in range(series_length + init_washout_length):
            for i in range(incremental_per_unit * speed_up):
                step += 1
                tau_value = gen_history[np.mod(step, gen_history_length)]
                new_value = old_value + (0.2 * tau_value/(1.0+np.power(tau_value, 10))-0.1*old_value)/incremental_per_unit
                gen_history[np.mod(step, gen_history_length)] = old_value
                old_value = new_value
            if n >= init_washout_length:
                mgs[n - init_washout_length] = new_value

        mgs = np.tanh(np.subtract(mgs, 1.0))

        return mgs
