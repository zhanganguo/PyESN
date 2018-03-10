import numpy as np
import scipy.linalg
from weights_generator.reservoir_weights.ReservoirWeightsGenerator import ReservoirWeightsGenerator


class RandomReservoirWeightsGenerator(ReservoirWeightsGenerator):
    def __init__(self):
        ReservoirWeightsGenerator.__init__(self)

    def generate_reservoir_weights(self, esn):
        reservoir_dimension = esn.reservoir_dimension
        reservoir_weights = np.random.rand(reservoir_dimension, reservoir_dimension) - 0.5
        rhoW = max(abs(scipy.linalg.eig(reservoir_weights)[0]))
        reservoir_weights *= esn.spectral_radius / rhoW
        return reservoir_weights
