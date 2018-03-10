import ESN
from weights_generator.input_weights.RandomInputWeightsGenerator import RandomInputWeightsGenerator
from weights_generator.reservoir_weights.RandomReservoirWeightsGenerator import RandomReservoirWeightsGenerator
from weights_generator.output_weights.RandomOutputWeightsGenerator import RandomOutputWeightsGenerator
from weights_generator.feedback_weights.RandomFeedbackWeightsGenerator import RandomFeedbackWeightsGenerator
from data.MackeyGlassSeries import MackeyGlassSeries
from pylab import *
from train.LinearRegressionTrainer import LinearRegressionTrainer


if __name__ == '__main__':
    esn = ESN.ESN(1, 400, 1)
    esn.generate_input_weights(input_weights_generator=RandomInputWeightsGenerator())
    esn.generate_reservoir_weights(reservoir_weights_generator=RandomReservoirWeightsGenerator())
    esn.generate_output_weights(output_weights_generator=RandomOutputWeightsGenerator())
    esn.generate_feedback_weights(feedback_weights_generator=RandomFeedbackWeightsGenerator())

    train_data_x = np.ones((1000, 1)) * 0.2
    train_data_y = MackeyGlassSeries().generate_data(16, 1000)
    test_data_x = np.ones((1000, 1)) * 0.2
    test_data_y = MackeyGlassSeries().generate_data(16, 1000)

    data = {'train': {'x': train_data_x, 'y': train_data_y},
            'test':  {'x': test_data_x,  'y': test_data_y}}

    esn.train(data=data, trainer=LinearRegressionTrainer())

