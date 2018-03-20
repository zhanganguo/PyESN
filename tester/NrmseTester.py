from pylab import *
from tester.Tester import Tester
from ESN import ESN


class NrmseTester(Tester):
    def __init__(self, test_mode):
        Tester.__init__(self, test_mode)

    def test(self, esn, data):
        test_run_length = len(data) - 1

        Y = zeros((esn.output_dimension, test_run_length))
        u = data[0]

        for t in range(test_run_length):
            reservoir_input = dot(esn.input_weights, vstack((1, u))) + dot(esn.reservoir_weights, esn.reservoir_state)
            reservoir_output = tanh(reservoir_input)

            esn.reservoir_state = (1 - esn.leaky_rate) * esn.reservoir_state + esn.leaky_rate * reservoir_output

            if esn.connection_mode == ESN.CONNECTION_RESERVOIR_AND_INPUT:
                y = dot(esn.output_weights, vstack((1, u, esn.reservoir_state)))
            elif esn.connection_mode == ESN.CONNECTION_RESERVOIR:
                y = dot(esn.output_weights, vstack((1, esn.reservoir_state)))
            else:
                raise Exception(ESN.CONNECTION_MODE_ERROR_TIP)

            Y[:, t] = y
            if self.test_mode == NrmseTester.TEST_MODE_GENERATIVE:
                u = y
            elif self.test_mode == NrmseTester.TEST_MODE_PREDICTION:
                u = data[t + 1]
            else:
                raise Exception(ESN.TASK_MODE_ERROR_TIP)

        mse_for_each_t = square(data[1: test_run_length+1] - Y[0, 0: test_run_length])
        NRMSE = sqrt(mean(mse_for_each_t) / var(data[0: test_run_length]))
        print('NRMSE = ' + str(NRMSE))

        figure(100).clear()
        plot(data[1: test_run_length + 1], color='blue')
        plot(Y[0, 0: test_run_length],     color='red', linestyle='--')
        show()

