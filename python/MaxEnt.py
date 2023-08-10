
def binnarize(point_process, index, tmax, tmin = -500, dt = 10):
    if int(tmax/dt) > 0:
        return [np.histogram(point_process[index], bins = int(tmax/dt), range = (tmin, tmax))[0]]
    else: return [[]]

binned = point_process.apply(lambda x: binnarize(x*1000, 'time', x.duration*1000),axis=1)
    

per_neuron = binned.reset_index().groupby('unit').apply(lambda x: np.hstack(x.time))
    

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.heatmap(pd.DataFrame(np.vstack(per_neuron)))



from sklearn.covariance import oas

oas()





cov



