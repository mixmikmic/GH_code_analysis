import axelrod as axl
import random
get_ipython().magic('matplotlib inline')

strategies = [s() for s in axl.basic_strategies]
strategies.append(axl.Random())
tournament = axl.Tournament(strategies)
results = tournament.play()

results.ranked_names

plot = axl.Plot(results)
plot.boxplot();

