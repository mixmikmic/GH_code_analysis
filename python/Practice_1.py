get_ipython().magic('matplotlib inline')
import numpy
import pandas
import matplotlib.pyplot as plt

import os, sys
sys.path.append('../code')

from utils import straight_tracks_generator
# add random state
events = straight_tracks_generator(n_events=10, n_tracks=3, n_noise=10, sigma=0.1, random_state=43)

events.head()

# take only the first event
event = events[events.EventID == 0]

from utils import plot_straight_tracks

plot_straight_tracks(event, labels=None)
plt.xlim(-0.5, 9.5);

X = event.X.values
y = event.y.values

from TemplateMatching import SimpleTemplateMatching

stm = SimpleTemplateMatching(8, 1.0)
stm.fit(X, y)

params = stm.tracks_params_
labels = stm.labels_

labels

plot_straight_tracks(event, labels)
plt.xlim(-0.5, 9.5);

from sklearn.linear_model import RANSACRegressor
from RANSAC import RANSACTracker

sk_ransac = RANSACRegressor(min_samples=3, max_trials=1000, residual_threshold=1.)
ransac = RANSACTracker(3, 3, regressor=sk_ransac)
ransac.fit(X, y)

labels = ransac.labels_

labels

plot_straight_tracks(event, labels)
plt.xlim(-0.5, 9.5);

from retina import Retina2DTrackerTwo

rt = Retina2DTrackerTwo(n_tracks=3, residuals_threshold=1.0, sigma=0.2, min_hits=3)
rt.fit(X, y)
labels = rt.labels_

plot_straight_tracks(event, labels)
plt.xlim(-0.5, 9.5);

from DenbyPeterson import DenbyPeterson

dp = DenbyPeterson(n_iter=10, 
                   cos_degree=1, 
                   alpha=0.00, 
                   delta=0.00, 
                   temperature=1, 
                   temperature_decay_rate=1., 
                   max_cos=-0.9,
                   state_threshold=0.5,
                   min_hits=3,
                   save_stages=True)
dp.fit(X, y)

satets = dp.states_
states_after_cut = dp.states_after_cut_
labels = dp.labels_


energy_stages = dp.energy_stages_

plt.plot(energy_stages);

from DenbyPeterson import plot_neural_net

plot_neural_net(X, y, states_after_cut, 0.5)

plot_straight_tracks(event, labels)
plt.xlim(-0.5, 9.5);

from metrics import HitsMatchingEfficiency

hme = HitsMatchingEfficiency()
hme.fit(event, labels)

# after fit we have all quality values as the following properties:
print 'tracks eff:\t', hme.efficiencies_
print 'average eff:\t', hme.avg_efficiency_
print 'reconstruction eff:\t', hme.reconstruction_efficiency_
print 'ghost rate:\t', hme.ghost_rate_
print 'clone rate:\t', hme.clone_rate_

from metrics import ParameterMatchingEfficiency

pme = ParameterMatchingEfficiency(delta_k = .2, delta_b=1.)
pme.fit(event, labels)

# after fit we have all quality values as the following properties
print 'reconstruction eff:\t', pme.reconstruction_efficiency_
print 'ghost rate:\t', pme.ghost_rate_
print 'clone rate:\t', pme.clone_rate_

from copy import copy
from metrics import HitsMatchingEfficiency, ParameterMatchingEfficiency

def get_quality_meatrics(events, model):
    
    results = pandas.DataFrame(columns=['EventID', 'HmAvgEff',
                                        'HmRecoEff', 'HmGhostRate', 'HmCloneRate', 
                                        'PmRecoEff', 'PmGhostRate', 'PmCloneRate'])

    for event_id in numpy.unique(events.EventID.values):
        event = events[events.EventID == event_id]
        x = event.X.values
        y = event.y.values
        
        model_object = copy(model)
        model_object.fit(x, y)
        labels = model_object.labels_

        hme = HitsMatchingEfficiency()
        hme.fit(event, labels)

        pme = ParameterMatchingEfficiency(delta_k = 0.4, delta_b=2.)
        pme.fit(event, labels)

        results.loc[len(results)] = [event_id, hme.avg_efficiency_, 
                                     hme.reconstruction_efficiency_, hme.ghost_rate_, hme.clone_rate_, 
                                     pme.reconstruction_efficiency_, pme.ghost_rate_, pme.clone_rate_]
        
    return results

results = get_quality_meatrics(events, model=stm)

results.head()

results.mean(axis=0)

def plot_report(report, x_column):
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 2)
    plt.plot(report[x_column].values, report['PmRecoEff'].values, linewidth=4, label='Reco Eff')
    plt.plot(report[x_column].values, report['PmGhostRate'].values, linewidth=4, label='Ghost Rate')
    plt.plot(report[x_column].values, report['PmCloneRate'].values, linewidth=4, label='Clone Rate')
    plt.legend(loc='best', prop={'size':12})
    plt.xlabel(x_column, size=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title('Parameters Matching', size=12)
    
    plt.subplot(1, 2, 1)
    plt.plot(report[x_column].values, report['HmRecoEff'].values, linewidth=4, label='Reco Eff')
    plt.plot(report[x_column].values, report['HmGhostRate'].values, linewidth=4, label='Ghost Rate')
    plt.plot(report[x_column].values, report['HmCloneRate'].values, linewidth=4, label='Clone Rate')
    plt.legend(loc='best', prop={'size':12})
    plt.xlabel(x_column, size=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title('Hits Matching', size=12)
    
    plt.show()

events = straight_tracks_generator(n_events=100, n_tracks=3, n_noise=10, sigma=0.5)

stm = SimpleTemplateMatching(8, 0.1)

sk_ransac = RANSACRegressor(min_samples=3, max_trials=100, residual_threshold=0.5)
ransac = RANSACTracker(n_tracks=5, min_hits=2, regressor=sk_ransac)

rt = Retina2DTrackerTwo(n_tracks=5, residuals_threshold=0.02, sigma=0.1, min_hits=3)

dp = DenbyPeterson(n_iter=10, 
                   cos_degree=1, 
                   alpha=10.00, 
                   delta=10.00, 
                   temperature=10, 
                   temperature_decay_rate=1., 
                   max_cos=-0.9,
                   state_threshold=0.5,
                   min_hits=3,
                   save_stages=True)

get_ipython().run_cell_magic('time', '', 'results_stm = get_quality_meatrics(events, model=stm)\nresults_ransac = get_quality_meatrics(events, model=ransac)\nresults_rt = get_quality_meatrics(events, model=rt)\nresults_dp = get_quality_meatrics(events, model=dp)')

columns=['HmRecoEff', 'HmGhostRate', 'HmCloneRate', 'PmRecoEff', 'PmGhostRate', 'PmCloneRate']
total_report = pandas.DataFrame(columns=columns, index=['stm', 'ransac', 'rt', 'dp'])

total_report.loc['stm'] = list(results_stm[columns].mean(axis=0).values)
total_report.loc['ransac'] = list(results_ransac[columns].mean(axis=0).values)
total_report.loc['rt'] = list(results_rt[columns].mean(axis=0).values)
total_report.loc['dp'] = list(results_dp[columns].mean(axis=0).values)
total_report

columns=['sigma', 'HmRecoEff', 'HmGhostRate', 'HmCloneRate', 'PmRecoEff', 'PmGhostRate', 'PmCloneRate']
report = pandas.DataFrame(columns=columns)

sigmas = [0.1, 0.5, 1., 2., 3., 4., 5.]

for sigma in sigmas:
    events = straight_tracks_generator(n_events=100, n_tracks=3, n_noise=10, sigma=sigma)
    stm = SimpleTemplateMatching(8, 3*sigma)
    results_stm = get_quality_meatrics(events, model=stm)
    
    report.loc[len(report)] = [sigma] + list(results_stm[columns[1:]].mean(axis=0).values)

report

plot_report(report, 'sigma')

columns=['NTracks', 'HmRecoEff', 'HmGhostRate', 'HmCloneRate', 'PmRecoEff', 'PmGhostRate', 'PmCloneRate']
report = pandas.DataFrame(columns=columns)

ns_tracks = [1, 2, 3, 4, 5]

for n_tracks in ns_tracks:
    events = straight_tracks_generator(n_events=100, n_tracks=n_tracks, n_noise=10, sigma=0.5)
    stm = SimpleTemplateMatching(8, 0.99)
    results_stm = get_quality_meatrics(events, model=stm)
    
    report.loc[len(report)] = [n_tracks] + list(results_stm[columns[1:]].mean(axis=0).values)

report

plot_report(report, 'NTracks')

columns=['NNoise', 'HmRecoEff', 'HmGhostRate', 'HmCloneRate', 'PmRecoEff', 'PmGhostRate', 'PmCloneRate']
report = pandas.DataFrame(columns=columns)

ns_noise = [0, 5, 10, 20, 30, 40, 50]

for n_noise in ns_noise:
    events = straight_tracks_generator(n_events=100, n_tracks=3, n_noise=n_noise, sigma=0.5)
    stm = SimpleTemplateMatching(8, 0.99)
    results_stm = get_quality_meatrics(events, model=stm)
    
    report.loc[len(report)] = [n_noise] + list(results_stm[columns[1:]].mean(axis=0).values)

report

plot_report(report, 'NNoise')

class LinearHoughModel(object):
    def __init__(self, k_params=(-2, 2, 0.1), b_params=(-10, 10, 1), min_hits=4):
        self.labels_ = None
                
    def fit(self, x, y):
        # Get labels of the hits
        self.labels_ = numpy.random.randint(-1, 3, len(x))

events = straight_tracks_generator(n_events=10, 
                                   n_tracks=3, 
                                   n_noise=5, 
                                   sigma=0.01, 
                                   x_range=(0, 10, 1), 
                                   intersection=False, 
                                   y_range=(0, 10, 1))

event = events[events.EventID == 0]
X = event.X.values
y = event.y.values

lh = LinearHoughModel()
lh.fit(X, y)
labels = lh.labels_

plot_straight_tracks(event, labels)
plt.xlim(-0.5, 9.5);

