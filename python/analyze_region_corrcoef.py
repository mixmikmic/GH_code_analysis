get_ipython().magic('pylab nbagg')
from tvb.simulator.lab import *
import tvb.analyzers.correlation_coefficient as corr_coeff
from tvb.datatypes.time_series import TimeSeriesRegion

# neural mass model parameters
pars = {'a': 1.05,
        'b': -1.,
        'c': 0.0,
        'd': 0.1,
        'e': 0.0,
        'f': 1 / 3.,
        'g': 1.0,
        'alpha': 1.0,
        'beta': 0.2,
        'tau': 1.25,
        'gamma': -1.0}

# sampling frequency
sfreq = 2048.0

sim = simulator.Simulator(
    model=models.Generic2dOscillator(**pars),
    connectivity=connectivity.Connectivity(load_default=True, speed=4.0),
    coupling=coupling.Linear(a=0.033),
    integrator=integrators.HeunStochastic(dt=0.06103515625, noise=noise.Additive(nsig=numpy.array([2 ** -10, ]))),
    monitors=(monitors.TemporalAverage(period=1e3 / sfreq),
              monitors.ProgressLogger(period=2e3)),
    simulation_length=16e3
).configure()

(tavg_time, tavg_samples), _ = sim.run()

tsr = TimeSeriesRegion(connectivity=sim.connectivity,
                       data=tavg_samples,
                       sample_period=sim.monitors[0].period)
tsr.configure()

corrcoeff_analyser = corr_coeff.CorrelationCoefficient(time_series=tsr)
corrcoeff_data = corrcoeff_analyser.evaluate()
corrcoeff_data.configure()
FC = corrcoeff_data.array_data[..., 0, 0]

plot_tri_matrix(FC,
                cmap=pyplot.cm.RdYlBu_r, 
                node_labels= sim.connectivity.region_labels,
                size=[10., 10.],
                color_anchor=(-1.0, 1.0));



