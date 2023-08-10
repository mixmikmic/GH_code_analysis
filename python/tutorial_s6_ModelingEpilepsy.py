get_ipython().magic('pylab nbagg')
from tvb.simulator.lab import *

from tvb.simulator.plot.phase_plane_interactive import PhasePlaneInteractive

# Create an Epileptor model instance
epileptor = models.Epileptor()

# Open the phase plane tool with the epileptor model
ppi_fig = PhasePlaneInteractive(model=epileptor)
ppi_fig.show()

epileptors = models.Epileptor(Ks=-0.2, Kf=0.1, r=0.00015)
epileptors.x0 = np.ones((76))*-2.4
epileptors.x0[[62, 47, 40]] = np.ones((3))*-1.6
epileptors.x0[[69, 72]] = np.ones((2))*-1.8

con = connectivity.Connectivity(load_default=True)

coupl = coupling.Difference(a=1.)

hiss = noise.Additive(nsig = numpy.array([0., 0., 0., 0.0003, 0.0003, 0.]))
heunint = integrators.HeunStochastic(dt=0.05, noise=hiss)

# load the default region mapping
rm = region_mapping.RegionMapping(load_default=True)

#Initialise some Monitors with period in physical time
mon_tavg = monitors.TemporalAverage(period=1.)
mon_EEG = monitors.EEG(load_default=True,
                       region_mapping=rm,
                       period=1.) 
mon_SEEG = monitors.iEEG(load_default=True,
                         region_mapping=rm,
                         period=1.)

#Bundle them
what_to_watch = (mon_tavg, mon_EEG, mon_SEEG)

#Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim = simulator.Simulator(model=epileptors, connectivity=con,
                          coupling=coupl, 
                          integrator=heunint, monitors=what_to_watch)

sim.configure()

(ttavg, tavg), (teeg, eeg), (tseeg, seeg) = sim.run(simulation_length=10000)

# Normalize the time series to have nice plots
tavg /= (np.max(tavg,0) - np.min(tavg,0 ))
eeg /= (np.max(eeg,0) - np.min(eeg,0 ))
eeg -= np.mean(eeg, 0)
seeg /= (np.max(seeg,0) - np.min(seeg, 0))
seeg -= np.mean(seeg, 0)


#Plot raw time series
figure(figsize=(10,10))
plot(ttavg[:], tavg[:, 0, :, 0] + np.r_[:76], 'r')
title("Epileptors time series")
show()

figure(figsize=(10,10))
plot(teeg[:], 10*eeg[:, 0, :, 0] + np.r_[:65])
yticks(np.r_[:75], mon_SEEG.sensors.labels[:65])
title("EEG")

figure(figsize=(10,10))
plot(tseeg[:], seeg[:, 0, :75, 0] + np.r_[:75])
yticks(np.r_[:75], mon_SEEG.sensors.labels[:75])
title("SEEG")

con = connectivity.Connectivity(load_default=True)
con.weights[[ 47, 40]] = 0.
con.weights[:, [47, 40]] = 0.

# we plot only the right hemisphere
# the lines and columns set to 0 are clearly visible
figure()
imshow(con.weights[38:, 38:], interpolation='nearest', cmap='binary')
colorbar()
show()

coupl = coupling.Difference(a=1.)
hiss = noise.Additive(nsig = numpy.array([0., 0., 0., 0.0003, 0.0003, 0.]))
heunint = integrators.HeunStochastic(dt=0.05, noise=hiss)
mon_tavg = monitors.TemporalAverage(period=1.)

epileptors = models.Epileptor(Ks=-0.2, Kf=0.1, r=0.00015)
epileptors.x0 = np.ones((76))*-2.4
epileptors.x0[[69, 72]] = np.ones((2))*-1.8
sim = simulator.Simulator(model=epileptors, connectivity=con,
                          coupling=coupl, 
                          integrator=heunint, monitors=mon_tavg)

sim.configure();

(ttavg, tavg), = sim.run(simulation_length=10000)

# normalize the time series
tavg /= (np.max(tavg,0) - np.min(tavg,0 ))

figure(figsize=(10,10))
plot(ttavg[:], tavg[:, 0, :, 0] + np.r_[:76], 'r')
title("Epileptors time series")
show()

epileptors = models.Epileptor(Ks=-0.2, Kf=0.1, r=0.00035)
epileptors.x0 = np.ones((76))*-2.1
con = connectivity.Connectivity(load_default=True)
coupl = coupling.Difference(a=0.)
heunint = integrators.HeunDeterministic(dt=0.05)
mon_tavg = monitors.TemporalAverage(period=1.)

# Weighting for regions to receive stimuli.
weighting = numpy.zeros((76))
weighting[[69, 72]] = numpy.array([2.])

eqn_t = equations.PulseTrain()
eqn_t.parameters["T"] = 10000.0
eqn_t.parameters["onset"] = 2500.0
eqn_t.parameters["tau"] = 50.0
eqn_t.parameters["amp"] = 20.0
stimulus = patterns.StimuliRegion(temporal = eqn_t,
                                  connectivity = con, 
                                  weight = weighting)

#Configure space and time
stimulus.configure_space()
stimulus.configure_time(numpy.arange(0., 3000., heunint.dt))

#And take a look
plot_pattern(stimulus)
show()

#Bundle them
what_to_watch = (mon_tavg, )
#Initialise Simulator -- Model, Connectivity, Integrator, Monitors, and stimulus.
sim = simulator.Simulator(model=epileptors, connectivity=con,
                          coupling=coupl, 
                          integrator=heunint, monitors=what_to_watch, 
                          stimulus=stimulus)

sim.configure()

(t, tavg), = sim.run(simulation_length=10000)

# Normalize the time series to have nice plots
tavg /= (np.max(tavg,0) - np.min(tavg,0 ))

#Plot raw time series
figure(figsize=(10,10))
plot(t[:], tavg[:, 0, :, 0] + np.r_[:76], 'r')
title("Epileptors time series")

#Show them
show()

