from pylab import *


# plot the activation of all neurons in the reservoir during 1 epoch
def plot_activity(reservoir_state):
    figure(1).clear()
    hist(reservoir_state.ravel(), bins=200)
    xlim(-1, +1)
    xlabel('neurons outputs')
    ylabel('number of neurons')
    # compute some characteristics of the distribution
    mean = str(round(reservoir_state.mean(), 2))
    med = str(round(median(reservoir_state), 2))
    min = str(round(reservoir_state.min(), 2))
    max = str(round(reservoir_state.max(), 2))
    std_dev = str(round(reservoir_state.std(), 2))
    title('Spatio-temporal distribution of the reservoir ' +
          'mean = ' + mean + ' median = ' + med + ' min = ' +
          min + ' max = ' + max + ' std_dev = ' + std_dev)


def plot_neuron_activity(activity):
    figure( 84).clear()
    hist(activity.ravel(), bins=200)
    xlim(-1, +1)
    xlabel('neuron outputs')
    ylabel('number of timesteps')
    title('The output distribution of a single randomly chosen neuron')

