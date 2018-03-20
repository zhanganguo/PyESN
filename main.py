from numpy import *
from ESN import ESN
from weights_generator.input_weights.RandomInputWeightsGenerator import RandomInputWeightsGenerator
from weights_generator.reservoir_weights.RandomReservoirWeightsGenerator import RandomReservoirWeightsGenerator
from trainer.LinearRegressionTrainer import LinearRegressionTrainer
from tester.NrmseTester import NrmseTester
from data.MackeyGlassSeries import MackeyGlassSeries


MackeyGlassSeries().generate_data(tau=16, series_length=10000)
data = loadtxt('data/MackeyGlass_t16.txt')

esn = ESN(input_dimension=1, reservoir_dimension=500, output_dimension=1)
esn.configure(leaky_rate=0.3, spectral_radius=0.95, input_scale=1.0)

esn.generate_input_weights(input_weights_generator=RandomInputWeightsGenerator())
esn.generate_reservoir_weights(reservoir_weights_generator=RandomReservoirWeightsGenerator())

esn.train(data=data[0: 2000], trainer=LinearRegressionTrainer())
esn.test(data=data[2001: 4000], tester=NrmseTester(test_mode=NrmseTester.TEST_MODE_GENERATIVE))
