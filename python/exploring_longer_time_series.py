get_ipython().magic('pylab nbagg')
import os
from tvb.simulator.lab import *

sim = simulator.Simulator(
    connectivity=connectivity.Connectivity(load_default=True, speed=1.0),
    coupling=coupling.Linear(a=2e-4),
    integrator=integrators.EulerStochastic(dt=10.0),
    model=models.Linear(gamma=-1e-2),
    monitors=monitors.Raw(),
    simulation_length=60e3
).configure()

(time, data), = sim.run()

figure()
plot(time/1e3, data[:, 0, :, 0], 'k', alpha=0.1);
xlabel('Time (s)')

tsr = time_series.TimeSeriesRegion(
    data=data,
    connectivity=sim.connectivity,
    sample_period=sim.monitors[0].period / 1e3,
    sample_period_unit='s')
tsr.configure()
tsr

#Create and launch the interactive visualiser
import tvb.simulator.plot.timeseries_interactive as ts_int
tsi = ts_int.TimeSeriesInteractive(time_series=tsr)
tsi.configure()
tsi.show()

