import numpy as np
from weights_generator.output_weights.OutputWeightsGenerator import OutputWeightsGenerator
from ESN import ESN


class RandomOutputWeightsGenerator(OutputWeightsGenerator):
    def __init__(self):
        OutputWeightsGenerator.__init__(self)

    def generate_output_weights(self, esn):
        input_dimension = esn.input_dimension
        reservoir_dimension = esn.reservoir_dimension
        output_dimension = esn.output_dimension
        if esn.output_mode == ESN.OUTPUT_FULL:
            output_weights = np.zeros((output_dimension, input_dimension + reservoir_dimension))
        elif esn.output_mode == ESN.OUTPUT_RESERVOIR:
            output_weights = np.zeros((output_dimension, reservoir_dimension))
        else:
            raise Exception('ERROR: \'output_mode\' of ESN has not been set.')
        return output_weights
