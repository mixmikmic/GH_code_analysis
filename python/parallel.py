import random
import numpy as np
from matplotlib import pyplot as plt

# Create sorted random array and random element of that array; this just sets up the problem.
def rand_arr(n_elements=100000):
    
    random_list = [random.random() for i in np.arange(n_elements)]
    random_list.sort()
    return random_list

random_list = rand_arr()
new_number = random.random()

get_ipython().run_cell_magic('timeit', '', '# complete')

get_ipython().run_cell_magic('timeit', '', '# complete')

get_ipython().run_cell_magic('timeit', '', 'list_of_list_of_ran = [rand_arr() for i in range(100)]\n# complete')

get_ipython().run_cell_magic('timeit', '', '# complete')



