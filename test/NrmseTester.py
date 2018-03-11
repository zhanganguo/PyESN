import numpy as np
from test.Tester import Tester
from test.Tester_Config import *
from util.Util import Util


class NrmseTester(Tester):
    def __init__(self):
        Tester.__init__(self)

    def test(self, esn, data):
        x, y = Util.extract_data(data=data)
        data_length = x.shape[0]
        if data_length < INITIAL_RUN_LENGTH + TEST_RUN_LENGTH:
            raise Exception('The length of data is not sufficient! Required:', INITIAL_RUN_LENGTH + TEST_RUN_LENGTH)
        total_state = np.zeros(esn.input_dimension + esn.reservoir_dimension + esn.output_dimension, 1)
        weights_matrix = np.concatenate((esn.input_weights.T, esn.reservoir_weights.T, esn.feedback_weights.T),
                                        axis=0).T
        for t in range(INITIAL_RUN_LENGTH):
            u = x[t, :]
            total_state[0: esn.input_dimension] = u
            reservoir_state = np.tanh(np.matmul(weights_matrix, total_state))
            practical_output = np.matmul(esn.output_weights, np.append(u, reservoir_state))
            total_state = np.concatenate((np.array([u]).T, reservoir_state, np.array([practical_output]).T), axis=0)

        start_state = total_state

        for trial in range(TEST_TRIALS):
            total_state = start_state
            for t in range(TRIAL_SHIFT):
                index = INITIAL_RUN_LENGTH + t + (trial - 1) * TRIAL_SHIFT
                u = x[index, :]
                total_state[0: esn.input_dimension] = u
                reservoir_state = np.tanh(np.matmul(weights_matrix, total_state))
                practical_output = np.matmul(esn.output_weights, np.append(u, reservoir_state))
                total_state = np.concatenate((np.array([u]).T, reservoir_state, np.array([practical_output]).T), axis=0)

            for t in range(TEST_RUN_LENGTH):
                index = INITIAL_RUN_LENGTH + t + trial * TRIAL_SHIFT
                u = x[index, :]
                target_output = y[index, :]
                total_state[0: esn.input_dimension] = u
                reservoir_state = np.tanh(np.matmul(weights_matrix, total_state))
                practical_output = np.matmul(esn.output_weights, np.append(u, reservoir_state))
                total_state = np.concatenate((np.array([u]).T, reservoir_state, np.array([practical_output]).T), axis=0)
