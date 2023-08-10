get_ipython().magic('pylab nbagg')
from tvb.simulator.lab import *
from tvb.simulator.plot.phase_plane_interactive import PhasePlaneInteractive
from tvb.simulator.models.epileptorcodim3 import EpileptorCodim3
from tvb.simulator.models.epileptorcodim3 import EpileptorCodim3SlowMod
from mpl_toolkits.mplot3d import Axes3D

Epileptorcd3 = EpileptorCodim3()

Epileptorcd3.state_variable_range = {"x": array([-2.0, 2.0]),
                 "y": array([-2.0, 2.0]),
                 "z": array([-0.1, 0.3])}
ppi_fig = PhasePlaneInteractive(model=Epileptorcd3)
ppi_fig.show()
Epileptorcd3.state_variable_range = {"x": numpy.array([0.4, 0.6]),
                 "y": numpy.array([-0.1, 0.1]),
                 "z": numpy.array([0.0, 0.1])}

Epileptorcd3.variables_of_interest=['x', 'y', 'z']
sim = simulator.Simulator(
    model=Epileptorcd3,
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 10,
).configure()

(tavg_time, tavg_data), = sim.run()

figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
show()

fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

A = [0.2649, -0.05246, 0.2951]
B = [0.2688, 0.05363, 0.2914]
c = 0.001

sim = simulator.Simulator(
    model= EpileptorCodim3(variables_of_interest=['x', 'y', 'z'], mu1_start=-A[1], mu2_start=A[0], nu_start=A[2],mu1_stop=-B[1], mu2_stop=B[0], nu_stop=B[2], c=c),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 11,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

A = [0.3448,0.02285,0.2014]
B = [0.3351,0.07465,0.2053]
c=0.001 

sim = simulator.Simulator(
    model= EpileptorCodim3(variables_of_interest=['x', 'y', 'z'], mu1_start=-A[1], mu2_start=A[0], nu_start=A[2],mu1_stop=-B[1], mu2_stop=B[0], nu_stop=B[2], c=c),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 10,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

A = [0.2552,-0.0637,0.3014]
B = [0.3496,0.0795,0.1774]
c=0.0004 

sim = simulator.Simulator(
    model= EpileptorCodim3(variables_of_interest=['x', 'y', 'z'], mu1_start=-A[1], mu2_start=A[0], nu_start=A[2],mu1_stop=-B[1], mu2_stop=B[0], nu_stop=B[2], c=c),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 13,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

A = [0.3448,0.0228,0.2014]
B = [0.3118,0.0670,0.2415]
c=0.00005 

sim = simulator.Simulator(
    model= EpileptorCodim3(variables_of_interest=['x', 'y', 'z'], mu1_start=-A[1], mu2_start=A[0], nu_start=A[2],mu1_stop=-B[1], mu2_stop=B[0], nu_stop=B[2], c=c),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 14,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

A = [0.3131,-0.06743,0.2396]
B = [0.3163,0.06846,0.2351]
c=0.00004 

sim = simulator.Simulator(
    model= EpileptorCodim3(variables_of_interest=['x', 'y', 'z'], mu1_start=-A[1], mu2_start=A[0], nu_start=A[2],mu1_stop=-B[1], mu2_stop=B[0], nu_stop=B[2], c=c),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 15,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

A = [0.3216,0.0454,-0.2335]
B = [0.285,0.05855,-0.2745]
c=0.004 

sim = simulator.Simulator(
    model= EpileptorCodim3(variables_of_interest=['x', 'y', 'z'], mu1_start=-A[1], mu2_start=A[0], nu_start=A[2],mu1_stop=-B[1], mu2_stop=B[0], nu_stop=B[2], c=c),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 10,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

A = [0.1871,-0.02512,-0.3526]
B = [0.2081,-0.01412,-0.3413]
c=0.008

sim = simulator.Simulator(
    model= EpileptorCodim3(variables_of_interest=['x', 'y', 'z'], mu1_start=-A[1], mu2_start=A[0], nu_start=A[2],mu1_stop=-B[1], mu2_stop=B[0], nu_stop=B[2], c=c),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 10,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

A = [0.3216,0.0454,-0.2335]
B = [0.106,0.005238,-0.3857]
c=0.002 

sim = simulator.Simulator(
    model= EpileptorCodim3(variables_of_interest=['x', 'y', 'z'], mu1_start=-A[1], mu2_start=A[0], nu_start=A[2],mu1_stop=-B[1], mu2_stop=B[0], nu_stop=B[2], c=c),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 12,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

A = [0.04098,-0.07373,-0.391]
B = [-0.01301,-0.03242,-0.3985]
c=0.004 

sim = simulator.Simulator(
    model= EpileptorCodim3(variables_of_interest=['x', 'y', 'z'], mu1_start=-A[1], mu2_start=A[0], nu_start=A[2],mu1_stop=-B[1], mu2_stop=B[0], nu_stop=B[2], c=c),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 10,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

Epileptorcd3=EpileptorCodim3()

phi_start = -0.2
theta_start = 0.93
phi_stop = pi/4
theta_stop = 0.2

Epileptorcd3.R=0.4 #Default radius of the sphere
Epileptorcd3.mu2_start=Epileptorcd3.R*sin(theta_start)*cos(phi_start)
Epileptorcd3.mu1_start=-Epileptorcd3.R*sin(theta_start)*sin(phi_start)
Epileptorcd3.nu_start=Epileptorcd3.R*cos(theta_start)
Epileptorcd3.mu2_stop=Epileptorcd3.R*sin(theta_stop)*cos(phi_stop)
Epileptorcd3.mu1_stop=-Epileptorcd3.R*sin(theta_stop)*sin(phi_stop)
Epileptorcd3.nu_stop=Epileptorcd3.R*cos(theta_stop)
Epileptorcd3.c=0.001

Epileptorcd3.variables_of_interest=['x', 'y', 'z']
sim = simulator.Simulator(
    model=Epileptorcd3,
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 12,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

dstar=0.3

sim = simulator.Simulator(
    model=EpileptorCodim3(dstar=dstar, variables_of_interest=['x', 'y', 'z']),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 11,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

dstar=0.1

sim = simulator.Simulator(
    model=EpileptorCodim3(dstar=dstar, variables_of_interest=['x', 'y', 'z']),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 11,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

dstar=0

sim = simulator.Simulator(
    model=EpileptorCodim3(dstar=dstar, variables_of_interest=['x', 'y', 'z']),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 11,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

dstar=1

sim = simulator.Simulator(
    model=EpileptorCodim3(dstar=dstar, variables_of_interest=['x', 'y', 'z']),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 11,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

dstar=-0.3

sim = simulator.Simulator(
    model=EpileptorCodim3(dstar=dstar, variables_of_interest=['x', 'y', 'z']),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 14,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

dstar=0
modification=1

sim = simulator.Simulator(
    model=EpileptorCodim3(dstar=dstar, variables_of_interest=['x', 'y', 'z'], modification=modification),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 14,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 2, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
fig2 = figure()
ax = fig2.add_subplot(111, projection='3d')
ax.plot(tavg_data[:, 0, 0, 0], tavg_data[:, 1, 0, 0], tavg_data[:, 2, 0, 0])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
show()

#Unmodified
sim = simulator.Simulator(
    model=EpileptorCodim3(),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 10,
).configure()

#Modified
modification=1
sim2 = simulator.Simulator(
    model=EpileptorCodim3(modification=modification),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 10,
).configure()
(tavg_time, tavg_data), = sim.run()
(tavg_time2, tavg_data2), = sim2.run()

figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x (no modification)')
plot(tavg_time, tavg_data[:, 1, 0, 0], label='z (no modification)')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")

figure()
plot(tavg_time2, tavg_data2[:, 0, 0, 0], label='x (with modification)')
plot(tavg_time2, tavg_data2[:, 1, 0, 0], label='z (with modification)')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
show()

EpileptorCodim3_slowmod()

sim = simulator.Simulator(
    model= EpileptorCodim3_slowmod(),
    connectivity=connectivity.Connectivity(load_default=True),
    coupling=coupling.Linear(a=0.0152),
    integrator=integrators.HeunDeterministic(dt=2 ** -4),
    monitors=monitors.TemporalAverage(period=2 ** -2),
    simulation_length=2 ** 14,
).configure()
(tavg_time, tavg_data), = sim.run()
figure()
plot(tavg_time, tavg_data[:, 0, 0, 0], label='x')
plot(tavg_time, tavg_data[:, 1, 0, 0], label='z')
legend()
grid(True)
xlabel('Time (ms)')
ylabel("Temporal average")
show()

