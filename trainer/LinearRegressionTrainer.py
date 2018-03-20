from trainer.Trainer import Trainer
from pylab import *
from ESN import ESN


class LinearRegressionTrainer(Trainer):
    def __init__(self):
        Trainer.__init__(self)

    def train(self, esn, data):
        train_run_length = len(data) - 1

        if esn.connection_mode == ESN.CONNECTION_RESERVOIR_AND_INPUT:
            state_collected_matrix = zeros((1 + esn.input_dimension + esn.reservoir_dimension, train_run_length))
        elif esn.connection_mode == ESN.CONNECTION_RESERVOIR:
            state_collected_matrix = zeros((1 + esn.reservoir_dimension, train_run_length))
        else:
            raise Exception(ESN.CONNECTION_MODE_ERROR_TIP)

        target_output = data[None, 1: train_run_length + 1]

        esn.load_reservoir_state(zeros((esn.reservoir_dimension, 1)))

        for t in range(train_run_length):
            u = data[t]
            reservoir_input = dot(esn.input_weights, vstack((1, u))) + dot(esn.reservoir_weights, esn.reservoir_state)

            reservoir_output = tanh(reservoir_input)
            esn.reservoir_state = (1 - esn.leaky_rate) * esn.reservoir_state + esn.leaky_rate * reservoir_output

            if esn.connection_mode == ESN.CONNECTION_RESERVOIR_AND_INPUT:
                state_collected_matrix[:, t] = vstack((1, u, esn.reservoir_state))[:, 0]
            elif esn.connection_mode == ESN.CONNECTION_RESERVOIR:
                state_collected_matrix[:, t] = vstack((1, esn.reservoir_state))[:, 0]
            else:
                raise Exception(ESN.CONNECTION_MODE_ERROR_TIP)

        # use ridge regression (linear regression with regularization)
        if esn.connection_mode == ESN.CONNECTION_RESERVOIR_AND_INPUT:
            eye_dimension = 1 + esn.input_dimension + esn.reservoir_dimension
        elif esn.connection_mode == ESN.CONNECTION_RESERVOIR:
            eye_dimension = 1 + esn.reservoir_dimension
        else:
            raise Exception(ESN.CONNECTION_MODE_ERROR_TIP)
        regularization_coefficient = 1e-8
        esn.output_weights = dot(dot(target_output, state_collected_matrix.T), linalg.inv(dot(state_collected_matrix,
                                        state_collected_matrix.T) + regularization_coefficient * eye(eye_dimension)))

        # use pseudo inverse
        # esn.output_weights = dot(target_output, linalg.pinv(state_collected_matrix))











