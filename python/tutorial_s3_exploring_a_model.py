get_ipython().magic('pylab nbagg')
from tvb.simulator.lab import *
from tvb.simulator.plot.phase_plane_interactive import PhasePlaneInteractive

oscillator = models.Generic2dOscillator()

oscillator

ppi_fig = PhasePlaneInteractive(model=oscillator)
ppi_fig.show()

oscillator

heunstochint = integrators.HeunStochastic(dt=2**-5)

# examine integrator's attributes
heunstochint

heunstochint.noise

ppi_fig = PhasePlaneInteractive(model=oscillator, integrator=heunstochint)
ppi_fig.show()

heunstochint.noise

sim = simulator.Simulator(
    model = oscillator, 
    connectivity = connectivity.Connectivity(load_default=True),
    coupling = coupling.Linear(a=0.0152), 
    integrator = heunstochint, 
    monitors = monitors.TemporalAverage(period=2**-1),
    simulation_length=1e3,
).configure()

# run
(tavg_time, tavg_data), = sim.run()

# plot
figure()
plot(tavg_time, tavg_data[:, 0, :, 0], 'k', alpha=0.1)
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")

