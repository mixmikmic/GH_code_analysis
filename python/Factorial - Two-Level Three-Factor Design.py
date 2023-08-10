import pandas as pd
import numpy as np
from numpy.random import rand

inputs_labels = {'x1' : 'Length of specimen (mm)',
                 'x2' : 'Amplitude of load cycle (mm)',
                 'x3' : 'Load (g)'}

dat = [('x1',250,350),
       ('x2',8,10),
       ('x3',40,50)]

inputs_df = pd.DataFrame(dat,columns=['index','low','high'])
inputs_df = inputs_df.set_index(['index'])
inputs_df['label'] = inputs_df.index.map( lambda z : inputs_labels[z] )

inputs_df

inputs_df['average']      = inputs_df.apply( lambda z : ( z['high'] + z['low'])/2 , axis=1)
inputs_df['span']         = inputs_df.apply( lambda z : ( z['high'] - z['low'])/2 , axis=1)

inputs_df['encoded_low']  = inputs_df.apply( lambda z : ( z['low']  - z['average'] )/( z['span'] ), axis=1)
inputs_df['encoded_high'] = inputs_df.apply( lambda z : ( z['high'] - z['average'] )/( z['span'] ), axis=1)

inputs_df = inputs_df.drop(['average','span'],axis=1)

inputs_df

import itertools
encoded_inputs = list( itertools.product([-1,1],[-1,1],[-1,1]) )
encoded_inputs

results = [(-1, -1, -1, 674),
           ( 1, -1, -1, 3636),
           (-1,  1, -1, 170),
           ( 1,  1, -1, 1140),
           (-1, -1,  1, 292),
           ( 1, -1,  1, 2000),
           (-1,  1,  1, 90),
           (  1, 1,  1, 360)]

results_df = pd.DataFrame(results,columns=['x1','x2','x3','y'])
results_df['logy'] = results_df['y'].map( lambda z : np.log10(z) )
results_df

real_experiment = results_df

var_labels = []
for var in ['x1','x2','x3']:
    var_label = inputs_df.ix[var]['label']
    var_labels.append(var_label)
    real_experiment[var_label] = results_df.apply(
        lambda z : inputs_df.ix[var]['low'] if z[var]<0 else inputs_df.ix[var]['high'] , 
        axis=1)

print("The values of each real variable in the experiment:")
real_experiment[var_labels]

# Compute the mean effect of the factor on the response,
# conditioned on each variable
labels = ['x1','x2','x3']

main_effects = {}
for key in labels:
    
    effects = results_df.groupby(key)['logy'].mean()

    main_effects[key] = sum( [i*effects[i] for i in [-1,1]] )

main_effects

import itertools

twoway_labels = list(itertools.combinations(labels, 2))

twoway_effects = {}
for key in twoway_labels:
    
    effects = results_df.groupby(key)['logy'].mean()
    
    twoway_effects[key] = sum([ i*j*effects[i][j]/2 for i in [-1,1] for j in [-1,1] ])

    # This somewhat hairy one-liner takes the mean of a set of sum-differences
    #twoway_effects[key] = mean([  sum([ i*effects[i][j] for i in [-1,1] ]) for j in [-1,1]  ])

twoway_effects

import itertools

threeway_labels = list(itertools.combinations(labels, 3))

threeway_effects = {}
for key in threeway_labels:
    
    effects = results_df.groupby(key)['logy'].mean()
    
    threeway_effects[key] = sum([ i*j*k*effects[i][j][k]/4 for i in [-1,1] for j in [-1,1] for k in [-1,1] ])

threeway_effects

s = "yhat = "

s += "%0.3f "%(results_df['logy'].mean())

for i,k in enumerate(main_effects.keys()):
    if(main_effects[k]<0):
        s += "%0.3f x%d "%( main_effects[k]/2.0, i+1 )
    else:
        s += "+ %0.3f x%d "%( main_effects[k]/2.0, i+1 )

print(s)

sigmasquared = 0.0050
k = len(inputs_df.index)
Vmean = (sigmasquared)/(2**k)
Veffect = (4*sigmasquared)/(2**k)
print("Variance in mean: %0.6f"%(Vmean))
print("Variance in effects: %0.6f"%(Veffect))

print(np.sqrt(Vmean))
print(np.sqrt(Veffect))

unc_a_0 = np.sqrt(Vmean)
print(unc_a_0)



