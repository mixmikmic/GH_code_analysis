get_ipython().magic('pylab nbagg')
from tvb.simulator.lab import *

base_dt = 0.1
var_order_dt = 5.0

methods = [
    (integrators.EulerDeterministic, base_dt),
    (integrators.HeunDeterministic, 2*base_dt),
    (integrators.Dop853, var_order_dt),
    (integrators.Dopri5, var_order_dt),
    (integrators.RungeKutta4thOrderDeterministic, 4*base_dt),
    (integrators.VODE, var_order_dt),
]

conn  = connectivity.Connectivity(load_default=True)

figure(figsize=(15, 10))
for i, (method, dt) in enumerate(methods):
    numpy.random.seed(42)
    sim = simulator.Simulator(
        connectivity=conn,
        model=models.Generic2dOscillator(a=0.1),
        coupling=coupling.Linear(a=0.0),
        integrator=method(dt=dt),
        monitors=monitors.TemporalAverage(period=5.0),
        simulation_length=1000.0,
    ).configure()
    (t, raw), = sim.run()
    
    if i == 0:
        euler_raw = raw
    else:
        if raw.shape[0] != euler_raw.shape[0]:
            continue
        raw = abs(raw - euler_raw) / euler_raw.ptp() * 100.0
    
    subplot(3, 2, i + 1)
    plot(t, raw[:, 0, :, 0], 'k', alpha=0.1)
    if i > 0:
        ylabel('% diff')
        plot(t, raw[:, 0, :, 0].mean(axis=1), 'k', linewidth=3)
    title(method._ui_name)
    grid(True)

conn  = connectivity.Connectivity(load_default=True)

raws = []
names = []
for i, (method, dt) in enumerate(methods):
    numpy.random.seed(42)
    sim = simulator.Simulator(
        connectivity=conn,
        model=models.Generic2dOscillator(a=0.1),
        coupling=coupling.Linear(a=0.0),
        integrator=method(dt=dt),
        monitors=monitors.TemporalAverage(period=5.0),
        simulation_length=200.0,
    ).configure()
    (t, raw), = sim.run()
    raws.append(raw)
    names.append(method._ui_name)

n_raw = len(raws)
figure(figsize=(15, 15))
for i, (raw_i, name_i) in enumerate(zip(raws, names)):
    for j, (raw_j, name_j) in enumerate(zip(raws, names)):
        subplot(n_raw, n_raw, i*n_raw + j + 1)
        if raw_i.shape[0] != t.shape[0] or raw_i.shape[0] != raw_j.shape[0]:
            continue
        if i == j:
            plot(t, raw_i[:, 0, :, 0], 'k', alpha=0.1)
        else:
            semilogy(t, (abs(raw_i - raw_j) / raw_i.ptp())[:, 0, :, 0], 'k', alpha=0.1)
            ylim(exp(r_[-30, 0]))
        
        grid(True)
        if i==0:
            title(name_j)
        if j==0:
            ylabel(name_i)
    
    if i == 0:
        euler_raw = raw
    else:
        raw = abs(raw - euler_raw) / euler_raw.ptp() * 100.0

tight_layout()

dts = [float(10**e) for e in r_[-2:0:10j]]
print dts
conn  = connectivity.Connectivity(load_default=True)

raws = []
for i, dt in enumerate(dts):
    numpy.random.seed(42)
    sim = simulator.Simulator(
        connectivity=conn,
        model=models.Generic2dOscillator(a=0.1),
        coupling=coupling.Linear(a=0.0),
        integrator=integrators.VODE(dt=dt),
        monitors=monitors.TemporalAverage(period=1.0),
        simulation_length=1200.0,
    ).configure()
    (t, raw), = sim.run()
    t = t[:1000]
    raw = raw[:1000]
    raws.append(raw)
    
figure(figsize=(10, 10))
for i, dt in enumerate(dts):
    subplot(len(dts)/3, 3+1, i + 1)
    if i == 0:
        dat = raws[i]
    else:
        dat = log10((abs(raws[i] - raws[0]) / raws[0].ptp()))
    plot(t, dat[:, 0, :, 0], 'k', alpha=0.1)

