get_ipython().magic('pylab nbagg')
import sys
from tvb.simulator.lab import *
LOG = get_logger('demo')
import scipy.stats
from sklearn.decomposition import FastICA
import time
import utils

def run_sim(conn, cs, D, cv=3.0, dt=0.5, simlen=1e3):
    sim = simulator.Simulator(
        model=models.Generic2dOscillator(a=0.0),
        connectivity=conn,
        coupling=coupling.Linear(a=cs),
        integrator=integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig=array([D]))),
        monitors=monitors.TemporalAverage(period=5.0) # 200 Hz
    )
    sim.configure()
    (t, y), = sim.run(simulation_length=simlen)
    return t, y[:, 0, :, 0]

conn = connectivity.Connectivity(load_default=True)

tic = time.time()
t, y = run_sim(conn, 6e-2, 5e-4, simlen=10*60e3)
'simulation required %0.3f seconds.' % (time.time() - tic, )

cs = []
for i in range(int(t[-1]/1e3)):
    cs.append(corrcoef(y[(t>(i*1e3))*(t<(1e3*(i+1)))].T))
cs = array(cs)
cs.shape

cs_z = arctanh(cs)
for i in range(cs.shape[1]):
    cs_z[:, i, i] = 0.0
_, p = scipy.stats.ttest_1samp(cs, 0.0)

figure(figsize=(10, 4))
subplot(121), imshow(conn.weights, cmap='binary', interpolation='none')
subplot(122), imshow(log10(p)*(p < 0.05), cmap='gray', interpolation='none');
show()

figure()
plot(r_[1:len(cs)+1], [corrcoef(cs_i.ravel(), conn.weights.ravel())[0, 1] for cs_i in cs])
ylim([0, 0.5])
ylabel('FC-SC correlation')
xlabel('Time (s)');
show()

def plot_roi_corr_map(reg_name):
    roi = find(conn.ordered_labels==reg_name)[0]
    cs_m = cs[2:].mean(axis=0)
    rm = utils.cortex.region_mapping
    utils.multiview(cs_m[roi][rm], shaded=False, suptitle=reg_name, figsize=(10, 5))

for reg in 'lM1 rPFCVL rPCS'.split():
    plot_roi_corr_map(reg)

ica = FastICA(n_components=5, max_iter=250)
ica.fit(y[t>2e3])

for i, comp in enumerate(ica.components_[:3]):
    utils.multiview(comp[utils.cortex.region_mapping], shaded=False, 
                           suptitle='ICA %d' % (i, ), figsize=(10, 5))

