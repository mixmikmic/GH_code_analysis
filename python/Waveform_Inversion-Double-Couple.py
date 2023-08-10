get_ipython().run_line_magic('matplotlib', 'notebook')
import time
import os
import scipy
import numpy as num
import matplotlib.pyplot as plt
import plotly.plotly as py
from collections import OrderedDict

import utils_nb

from pyrocko import gf, trace
from pyrocko import moment_tensor as mtm
from pyrocko.gf import ws, LocalEngine, Target, DCSource
from pyrocko import util, pile, model, config, trace, io, pile, catalog

km = 1000.

component = 'Z'
f_low = 0.05  # Hz, for a lowpass filter
taper = trace.CosFader(xfade=2.0)  # Cosine taper, 2s fade

phase = 'P' # Phase to fit
tmin_fit = 15. # [s] to fit before synthetic phase onset (from GFStore)
tmax_fit = 35. # [s] ... after

bounds = OrderedDict([
    ('north_shift', (-20.*km, 20.*km)),
    ('east_shift', (-20.*km, 20.*km)),
    ('depth', (3.*km, 8.*km)),
    ('magnitude', (6.2, 6.4)),
    ('strike', (100., 140.)),
    ('dip', (40., 60.)),
    ('rake', (-100, -150.)),
    ('timeshift', (-20., 20.)),
    ])

# Download the instrument-corrected 2009 Aquila Earthquake data
data_path = utils_nb.download_dir('aquila_realdata/')
data = pile.make_pile([data_path])
traces = data.all()  # retrieves the raw waveform data as a 2D `numpy.array`.

for tr in traces:
    tr.lowpass(4, f_low)

store_id = 'global_2s_25km'
if not os.path.exists(store_id):
    ws.download_gf_store(site='kinherd', store_id=store_id)

engine = gf.LocalEngine(store_superdirs=['.']) # The Path to where the gf_store(s)
store = engine.get_store(store_id)  # Load the store.

tmin = util.str_to_time('2009-04-06 00:00:00')  # beginning time of query
tmax = util.str_to_time('2009-04-06 05:59:59')  # ending time of query
event = catalog.GlobalCMT().get_events(
    time_range=(tmin, tmax),
    magmin=6.)[0]

base_source = gf.MTSource.from_pyrocko_event(event)

stations_list = model.load_stations('data/aquila_realdata/stations_short.txt')
for s in stations_list:
    s.set_channels_by_name(*component.split())

targets=[]
for s in stations_list:
    target = Target(
            lat=s.lat,
            lon=s.lon,
            store_id=store_id,   # The gf-store to be used for this target,
            interpolation='multilinear',  # Interpolation method between GFStore nodes
            quantity='displacement',
            codes=s.nsl() + ('BH' + component,))
    targets.append(target)

source = gf.DCSource(
    lat=event.lat,
    lon=event.lon)

def update_source(params):
    s = source
    s.north_shift = float(params[0])
    s.east_shift = float(params[1])
    s.depth = float(params[2])
    s.magnitude = float(params[3])
    s.strike = float(params[4])
    s.dip = float(params[5])
    s.rake = float(params[6])
    s.time = float(event.time - params[7])
    return source

def process_trace(trace, tmin, tmax, lowpass=False, inplace=True):
    if lowpass:
        trace.lowpass(4, f_low)
    trace = trace.chop(tmin, tmax, inplace=inplace)
    trace.taper(taper)
    return trace

iiter = 0

def trace_fit(params, line=None):
    global iiter
    update_source(params)

    # Forward model synthetic seismograms
    response = engine.process(source, targets)
    syn_traces = response.pyrocko_traces()

    misfits = 0.
    norms = 0.

    for obs, syn, target in zip(traces, syn_traces, targets):
        syn_phs = store.t(phase, base_source, target)
        
        tmin = base_source.time + syn_phs - tmin_fit  # start before theor. arrival
        tmax = base_source.time + syn_phs + tmax_fit  # end after theor. arrival
        
        syn = process_trace(syn, tmin, tmax)
        obs = process_trace(obs, tmin, tmax, lowpass=False, inplace=True)

        misfits += num.sqrt(num.sum((obs.ydata - syn.ydata)**2))
        norms += num.sqrt(num.sum(obs.ydata**2))
    
    misfit = num.sqrt(misfits**2 / norms**2)
    
    iiter += 1

    if line:
        data = {
            'y': [misfit],
            'x': [iiter],
        }
        line.data_source.stream(data)

    return misfit

def solve():
    t = time.time()

    result = scipy.optimize.differential_evolution(
        trace_fit,
        args=[p],
        bounds=tuple(bounds.values()),
        maxiter=15000,
        tol=0.01,
        callback=update_plot)

    source = update_source(result.x)
    source.regularize()

    print("Time elapsed: %.1f s" % (time.time() - t))
    print("Best model:\n - Misfit %f" % trace_fit(result.x))
    print(source)
    return result, source

from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook()

f = figure(title='SciPy Optimisation Progress',
           x_axis_label='# Iteration',
           y_axis_label='Misfit',
           plot_width=800,
           plot_height=300)
p = f.scatter([], [])
show(f, notebook_handle=True)

def update_plot(*a, **ka):
    push_notebook()

# Start the optimisation
result, best_source = solve()

def plot_traces(result):
    nstations = len(stations_list)
    response = engine.process(source, targets)
    syn_traces = response.pyrocko_traces()

    fig, axes = plt.subplots(nstations, squeeze=True, sharex=True)
    fig.subplots_adjust(hspace=0)
    plt.setp([ax.get_xticklabels() for ax in axes[:-1]], visible=False)

    for istation, (obs, syn, target) in enumerate(zip(traces, syn_traces, targets)):
        ax = axes[istation]
        tp = store.t(phase, base_source, target)
        tp_onset = base_source.time + tp
        tmin = tp_onset - tmin_fit
        tmax = tp_onset + tmax_fit
        
        syn = process_trace(syn, tmin, tmax)
        obs = process_trace(obs, tmin, tmax, lowpass=False, inplace=False)
        
        s1 = ax.plot(obs.get_xdata(), obs.ydata, color='b')
        s2 = ax.plot(syn.get_xdata(), syn.ydata, color='r')
        s3 = ax.plot([tp_onset, tp_onset], [tr.ydata.min(), tr.ydata.max()], 'k-', lw=2)

        ax.text(-.2, 0.5, stations_list[istation].station,
                transform=ax.transAxes)
        ax.set_yticklabels([], visible=False)

    axes[-1].set_xlabel('Time [s]')
    plt.suptitle('Waveform fits for %s-Phase and component %s' % (phase, component))
    plt.legend(
        (s1[0], s2[0], s3[0]),
        ('Data', 'Synthetic','%s Phase-onset' % phase),
        loc='upper center',
        bbox_to_anchor=(0.5, -2.),
        fancybox=True, shadow=True, ncol=5)
    
    plt.show()

def plot_snuffler(result, source):
    engine = gf.get_engine()
    response = engine.process(source, targets)
    syn_traces = response.pyrocko_traces()
    obs_traces = []
    
    for obs, syn, target in zip(traces, syn_traces, targets):
        tp = store.t('P', base_source, target)
        tmin = base_source.time + tp - tmin_fit
        tmax = base_source.time + tp + tmax_fit

        syn = process_trace(syn, tmin, tmax)
        obs = process_trace(obs, tmin, tmax, lowpass=False, inplace=False)

        obs_traces.append(obs)

    trace.snuffle(obs_traces + syn_traces, stations=stations_list, events=events)

def plot_stations():
    import folium
    fmap = folium.Map(
        location=[best_source.lat, best_source.lon],
        tiles='Stamen Terrain',
        zoom_start=3)
    folium.Marker([best_source.lat, best_source.lon],
        popup=('2009 Aquila Earthquake'),
        icon=folium.Icon(color='red', icon='info-sign')).add_to(fmap)
                  
    for s in stations_list:
        folium.Marker([s.lat, s.lon],
                      popup='<b>%s</b></h4>' % s.station).add_to(fmap)
    fmap.add_child(folium.LatLngPopup())
    return fmap
    
plot_stations()

plot_traces(result)

plot_snuffler(result)

