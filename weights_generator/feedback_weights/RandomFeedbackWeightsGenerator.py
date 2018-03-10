from weights_generator.feedback_weights.FeedbackWeightsGenerator import FeedbackWeightsGenerator
import numpy as np


class RandomFeedbackWeightsGenerator(FeedbackWeightsGenerator):
    def __init__(self):
        FeedbackWeightsGenerator.__init__(self)

    def generate_feedback_weights(self, esn):
        reservoir_dimension = esn.reservoir_dimension
        output_dimension = esn.output_dimension
        feedback_weights = np.random.rand(reservoir_dimension, output_dimension) - 0.5
        return feedback_weights

