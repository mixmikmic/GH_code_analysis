get_ipython().magic('pylab nbagg')
from tvb.simulator.lab import *

rww = models.ReducedWongWang(a=0.27, w=1.0, I_o=0.3)
S = linspace(0, 1, 50).reshape((1, -1, 1))
C = S * 0.0
dS = rww.dfun(S, C)

figure()
plot(S.flat, dS.flat)

sim = simulator.Simulator(
    model=rww,
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.5 / 50.0),
    integrator=integrators.EulerStochastic(dt=1, noise=noise.Additive(nsig=1e-5)), 
    monitors=monitors.TemporalAverage(period=1.),
    simulation_length=5e3
).configure()

(time, data), = sim.run()

figure()
plot(time, data[:, 0, :, 0], 'k', alpha=0.1);

