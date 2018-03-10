# -*- coding: utf-8


class ESN:
    OUTPUT_FULL = 'output_full'
    OUTPUT_RESERVOIR = 'output_reservoir'

    def __init__(self, input_dimension, reservoir_dimension, output_dimension):
        self.input_dimension = input_dimension
        self.reservoir_dimension = reservoir_dimension
        self.output_dimension = output_dimension

        self.input_scale = 1
        self.connectivity_rate = 0.1
        self.spectral_radius = 0.8
        self.output_mode = ESN.OUTPUT_FULL

        self.input_weights = None
        self.reservoir_weights = None
        self.output_weights = None
        self.feedback_weights = None

    def configure(self, input_scale=1, connectivity_rate=0.1, spectral_radius=0.8, output_mode=OUTPUT_FULL):
        self.input_scale = input_scale
        self.connectivity_rate = connectivity_rate
        self.spectral_radius = spectral_radius
        self.output_mode = output_mode

    def generate_input_weights(self, input_weights_generator):
        self.input_weights = input_weights_generator.generate_input_weights(self)

    def generate_reservoir_weights(self, reservoir_weights_generator):
        self.reservoir_weights = reservoir_weights_generator.generate_reservoir_weights(self)

    def generate_output_weights(self, output_weights_generator):
        self.output_weights = output_weights_generator.generate_output_weights(self)

    def generate_feedback_weights(self, feedback_weights_generator):
        self.feedback_weights = feedback_weights_generator.generate_feedback_weights(self)

    def train(self, data, trainer):
        trainer.train(esn=self, data=data)

