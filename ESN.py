# -*- coding: utf-8
import numpy as np


class ESN:
    CONNECTION_RESERVOIR_AND_INPUT = 'both the reservoir and input nodes are connected to the output layer.'
    CONNECTION_RESERVOIR = 'only the reservoir nodes are connected to the output layer.'

    CONNECTION_MODE_ERROR_TIP = "ERROR: 'connection mode' is not set correctly."
    TASK_MODE_ERROR_TIP = "ERROR: 'task mode' is not set correctly."

    def __init__(self, input_dimension, reservoir_dimension, output_dimension):
        self.input_dimension = input_dimension
        self.reservoir_dimension = reservoir_dimension
        self.output_dimension = output_dimension

        self.input_scale = 1.0
        self.connectivity_rate = 0.1
        self.spectral_radius = 0.8
        self.connection_mode = ESN.CONNECTION_RESERVOIR_AND_INPUT
        self.feedback_scale = 1.0

        self.leaky_rate = 0.5

        self.input_weights = None
        self.reservoir_weights = None
        self.output_weights = None
        self.feedback_weights = None

        self.reservoir_state = np.zeros((self.reservoir_dimension, 1), dtype=float)

    def configure(self, input_scale=1, connectivity_rate=0.1, spectral_radius=0.8,
                  feedback_scale=1.0, leaky_rate=0,
                  connection_mode=CONNECTION_RESERVOIR_AND_INPUT):
        self.input_scale = input_scale
        self.connectivity_rate = connectivity_rate
        self.spectral_radius = spectral_radius
        self.connection_mode = connection_mode
        self.feedback_scale = feedback_scale
        self.leaky_rate = leaky_rate

    def generate_input_weights(self, input_weights_generator):
        self.input_weights = input_weights_generator.generate_input_weights(self)
        self.input_weights *= self.input_scale

    def generate_reservoir_weights(self, reservoir_weights_generator):
        self.reservoir_weights = reservoir_weights_generator.generate_reservoir_weights(self)

    def generate_feedback_weights(self, feedback_weights_generator):
        self.feedback_weights = feedback_weights_generator.generate_feedback_weights(self)
        self.feedback_weights *= self.feedback_scale

    def load_reservoir_state(self, reservoir_state):
        self.reservoir_state = reservoir_state

    def train(self, data, trainer):
        trainer.train(esn=self, data=data)

    def test(self, data, tester):
        tester.test(esn=self, data=data)

