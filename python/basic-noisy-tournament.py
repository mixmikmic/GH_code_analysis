import axelrod as axl
import random # To set a seed
get_ipython().magic('matplotlib inline')

random.seed(1)
strategies = [axl.Cooperator(), axl.Defector(),
              axl.TitForTat(), axl.Grudger(),
              axl.Random()]
noise = 0.1
tournament = axl.Tournament(strategies, noise=noise)
results = tournament.play()
plot = axl.Plot(results)
plot.boxplot();



