from train.Trainer import Trainer
from train.Trainer_Config import *
from util.Util import Util
import numpy as np


class LinearRegressionTrainer(Trainer):
    def __init__(self):
        Trainer.__init__(self)

    def train(self, esn, data):
        train_x, train_y = Util.extract_data(data=data)
        state_collect_matrix = np.zeros((TRAIN_RUN_LENGTH, esn.input_dimension + esn.reservoir_dimension))
        target_collect_matrix = np.zeros((TRAIN_RUN_LENGTH, esn.output_dimension))
        total_state = np.zeros((esn.input_dimension + esn.reservoir_dimension + esn.output_dimension, 1))
        sample_length = train_x.shape[0]
        for t in range(sample_length):
            u = train_x[t, :]
            target_output = train_y[t, :]
            total_state[0: esn.input_dimension] = u
            weights_matrix = np.concatenate((esn.input_weights.T, esn.reservoir_weights.T, esn.feedback_weights.T), axis=0).T
            reservoir_state = np.tanh(np.matmul(weights_matrix, total_state) + NOISE_LEVEL * 2.0 *
                                      (np.random.rand(esn.reservoir_dimension, 1) - 0.5))
            practical_output = np.matmul(esn.output_weights, np.append(u, reservoir_state))
            total_state = np.concatenate((np.array([u]).T, reservoir_state, np.array([practical_output]).T), axis=0)

            if (t >= INITIAL_RUN_LENGTH) and (t <= INITIAL_RUN_LENGTH + TRAIN_RUN_LENGTH):
                collect_index = t - INITIAL_RUN_LENGTH
                state_collect_matrix[collect_index, :] = np.append(u.T, reservoir_state.T)
                target_collect_matrix[collect_index, :] = target_output.T

            if t < INITIAL_RUN_LENGTH + TRAIN_RUN_LENGTH:
                total_state[esn.input_dimension + esn.reservoir_dimension - 1:
                            esn.input_dimension + esn.reservoir_dimension + esn.output_dimension - 1] = target_output.T

            if t == INITIAL_RUN_LENGTH + TRAIN_RUN_LENGTH:
                esn.output_weights = np.matmul(np.linalg.pinv(state_collect_matrix), target_collect_matrix).T




