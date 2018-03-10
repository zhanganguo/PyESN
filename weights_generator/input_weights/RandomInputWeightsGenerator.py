from weights_generator.input_weights.InputWeightsGenerator import InputWeightsGenerator
import numpy as np


class RandomInputWeightsGenerator(InputWeightsGenerator):
    def __init__(self):
        InputWeightsGenerator.__init__(self)

    def generate_input_weights(self, esn):
        input_dimension = esn.input_dimension
        reservoir_dimension = esn.reservoir_dimension
        input_weights = np.random.rand(reservoir_dimension, input_dimension) - 0.5
        return input_weights
