from weights_generator.input_weights.InputWeightsGenerator import InputWeightsGenerator
from numpy import *


class RandomInputWeightsGenerator(InputWeightsGenerator):
    def __init__(self):
        InputWeightsGenerator.__init__(self)

    def generate_input_weights(self, esn):
        input_dimension = esn.input_dimension
        reservoir_dimension = esn.reservoir_dimension
        input_scale = esn.input_scale
        input_weights = random.rand(reservoir_dimension, input_dimension + 1) - 0.5
        input_weights *= input_scale
        return input_weights
