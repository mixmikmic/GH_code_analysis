get_ipython().run_line_magic('pylab', 'inline')
import pysd
import numpy

from scipy import signal
x = np.arange(0,100,.1)
t_series = np.sin(x) + np.random.uniform(size=(len(x))) + .5 * np.sin(5*x)

frequencies, power = signal.welch(t_series, scaling='spectrum')
plt.plot(frequencies, power)

from scipy import signal
x = np.arange(0,100,.1)
t_series = np.random.uniform(size=(len(x)))

frequencies, power = signal.welch(t_series, scaling='spectrum')
plt.plot(frequencies, power)

plt.plot(t_series)

model = pysd.read_vensim('Floating_Goals.mdl')

def steady_state(time_ratio, initial_gap):
    initial_state = 0
    state_adjustment_time = 10
    return model.run(params={
        'Initial State of the System': initial_state,
        'Initial Goal': initial_state + initial_gap,
        'State Adjustment Time': state_adjustment_time,
        'Goal Adjustment Time': state_adjustment_time * time_ratio,
    },
                    return_timestamps=[100],
                    reload=True)['State of the System'].iloc[-1]

steady_state(1, 1)

time_ratio = np.logspace(-1,1,20)
initial_gap = np.arange(-10,10,1)
tr, ig = np.meshgrid(time_ratio, initial_gap)

vss = np.vectorize(steady_state)
ss = vss(tr, ig) 

plt.contour(tr, ig, ss)
plt.xlabel('Goal Adjustment Time / State Adjustment Time')
plt.ylabel('Initial Goal')
plt.colorbar(label='Steady State');

comp_model = pysd.read_vensim('Competing_Goals.mdl')
def comp_steady_state(time_ratio, initial_gap):
    initial_state = 0
    state_adjustment_time = 10
    return model.run(params={
        'Initial State of the System': initial_state,
        'Initial Goal': initial_state + initial_gap,
        'State Adjustment Time': state_adjustment_time,
        'Goal Adjustment Time': state_adjustment_time * time_ratio,
    },
                    return_timestamps=[100],
                    reload=True)['State of the System'].iloc[-1]

steady_state(1, 1)



