import json
import plotly
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()

filename = 'washington_0888'
name = '[5-(6-Amino-9H-purin-9-yl)tetrahydro-2-furanyl]methanol'

msdir = '/Users/simon/Dropbox/beer_analysis/fingerid-104-traindata/spectra_massbank/'
treedir = '/Users/simon/Dropbox/beer_analysis/fingerid-104-traindata/trees/'

with open(treedir + filename + '.json','r') as f:
    t = json.load(f)

ms2peaks = []
with open(msdir + filename + '.ms') as f:
    for line in f:
        line = line.rstrip()
        if line.startswith('>'):
            continue
        if len(line) > 0:
            # It's a peak
            tokens = line.split(' ')
            ms2peaks.append((float(tokens[0]),float(tokens[1])))

fragments = t['fragments']

parent = t['fragments'][0]
print "Parent: {}, {}".format(parent['mz'],parent['molecularFormula'])
print "Fragments:"
for f in t['fragments'][1:]:
    print f['mz'],f['molecularFormula'],f['relativeIntensity']
print "Losses:"
for l in t['losses']:
    print l['molecularFormula'],l['source'],l['target']

maxi = 0.0
for mz,intensity in ms2peaks:
    if intensity > maxi:
        maxi = intensity

data = []
for mz,intensity in ms2peaks:
    data.append(
        Scatter(
            x = [mz,mz],
            y = [0,intensity/maxi],
            mode = 'lines',
            line = dict(
                color = 'rgba(100,100,100,0.3)',
            ),
            showlegend = False,
        )
    )


# Find the matches
match_dict = {parent['molecularFormula']: -1}
for f in t['fragments'][1:]:
    fmz = f['mz']
    best_match = -1
    closest = 1000.0
    for i,(mz,_) in enumerate(ms2peaks):
        if abs(mz - fmz) < closest:
            best_match = i
            closest = abs(mz-fmz)
    match_dict[f['molecularFormula']] = best_match
    data.append(
        Scatter(
            x = [ms2peaks[best_match][0],ms2peaks[best_match][0]],
            y = [0,ms2peaks[best_match][1]/maxi],
            mode = 'lines',
#             line = dict(
#                 color = 'rgba(200,100,100,1.0)',
#             ),
            name = 'fragment: ' + f['molecularFormula']
        )
    )

for l in t['losses']:
    source_pos = match_dict[l['source']]
    target_pos = match_dict[l['target']]
    if source_pos == -1:
        source_mass = parent['mz']
    else:
        source_mass = ms2peaks[source_pos][0]
    target_mass = ms2peaks[target_pos][0]
    intensity = ms2peaks[target_pos][1]/maxi
    data.append(
        Scatter(
            x = [target_mass,source_mass],
            y = [intensity,intensity],
            mode = 'lines',
            name= 'loss: ' + l['molecularFormula'],
            line = dict(
                dash = 'dash',
            ),
        )
    )
data.append(
    Scatter(
        x = [parent['mz'],parent['mz']],
        y = [0,1],
        mode = 'lines',
        name = 'Parent: {}'.format(parent['molecularFormula']),
        line = dict(
            color = 'rgb(0,0,255)',
        )
    )
)
title = filename
if not name == None:
    title += '({})'.format(name)
layout = Layout(
    title = title,
)
plotly.offline.iplot({'data':data,'layout':layout})




