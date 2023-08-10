import pandas as pd
import lifelines
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')

data = lifelines.datasets.load_dd()

data.head()
data.tail()

from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()

# kaplan-meier 

# KaplanMeierFitter.fit(event_times, event_observed=None,
#                      timeline=None, label='KM-estimate',
#                      alpha=None)
"""Parameters:
  event_times: an array, or pd.Series, of length n of times that
         the death event occured at
  event_observed: an array, or pd.Series, of length n -- True if
        the death was observed, False if the event was lost
        (right-censored). Defaults all True if event_observed==None
  timeline: set the index of the survival curve to this postively increasing array.
  label: a string to name the column of the estimate.
  alpha: the alpha value in the confidence intervals.
         Overrides the initializing alpha for this call to fit only.

Returns:
  self, with new properties like 'survival_function_'
""" 

T = data["duration"]
C = data["observed"]
kmf.fit(T, event_observed=C)

kmf.survival_function_.plot()
plt.title('Survival function of political regimes');

kmf.plot()

kmf.median_

## A leader is elected there is a 50% chance he or she will be gone in 3 years.

ax = plt.subplot(111)

dem = (data["democracy"] == "Democracy")
kmf.fit(T[dem], event_observed=C[dem], label="Democratic Regimes")
kmf.plot(ax=ax, ci_force_lines=True)
kmf.fit(T[~dem], event_observed=C[~dem], label="Non-democratic Regimes")
kmf.plot(ax=ax, ci_force_lines=True)

## ci_force_lines : force the confidence intervals to be line plots

plt.ylim(0,1);
plt.title("Lifespans of different global regimes");

