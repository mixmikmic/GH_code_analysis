from src.riskga import PlayerPool
from src.geneticplayer import GeneticPlayer
import matplotlib.pyplot as plt
import os
get_ipython().magic('matplotlib inline')

def plot(ga):
    plot_args={'layout':(3, 8), 'figsize':(16,8)}
    ga.gene_df.drop('iteration', axis=1).hist(**plot_args)
    plt.show()
def run(ga, n):
    for i in range(n):
        ga.iteration()
        plot(ga)
    ga.save('risk.json')
    ga.save_log('risk.csv')

get_ipython().run_cell_magic('time', '', "if os.path.exists('risk.json'):\n    ga = PlayerPool.load(GeneticPlayer, 'risk.json',  ranking_iterations=16, pool_size=1500)\nelse:\n    ga = PlayerPool(GeneticPlayer, pool_size=1500, ranking_iterations=16)\nplot(ga)")

get_ipython().run_cell_magic('time', '', 'run(ga, 1)')

get_ipython().run_cell_magic('time', '', 'run(ga, 3)')

get_ipython().run_cell_magic('time', '', 'run(ga, 3)')

get_ipython().run_cell_magic('time', '', 'run(ga, 3)')

get_ipython().run_cell_magic('time', '', 'run(ga, 30)')

get_ipython().run_cell_magic('time', '', 'run(ga, 30)')

get_ipython().run_cell_magic('time', '', 'run(ga, 30)')

