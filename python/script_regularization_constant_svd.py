from os import path
import numpy as np
import matplotlib.pyplot as plt

import sys
parent_path = path.abspath('..')
sys.path.insert(0, parent_path)
import dfFunctions
from utils import rmse
import recommender as re


path = parent_path + '/movielens/ml-1m/ratings.dat'
df = dfFunctions.load_dataframe(path)
model = re.SVDmodel(df, 'user', 'item', 'rating', 'svd')

learning_rate = 0.001
momentum_factor = 0.9
num_steps = 9000
batch_size = 700
dimension = 12

all_constants = np.random.random_sample([300])
results = []
times = []

for regularizer_constant in all_constants:
    print("\nregularizer constant = {}".format(regularizer_constant))
    model.training(dimension,
               regularizer_constant,
               learning_rate,
               momentum_factor,
               batch_size,
               num_steps,
               False)
    users, items, rates = model.test_batches.get_batch()
    predicted_ratings = model.prediction(users,items)
    result = rmse(predicted_ratings, rates)
    results.append(result)
    times.append(model.duration)

all_constants = list(all_constants)
best_result = min(list(zip(results,all_constants,times)))
result_string = """In an experiment with 300 random constants
the best  regularizer_constant is {0} with error {1}.
Using this constant the training will take {2} seconds""".format(
                                                             best_result[1],
                                                             best_result[0],
                                                             best_result[2])
print(result_string)

print(np.mean(results),np.std(results))

print(np.mean(times),np.std(times))

aggregate = list(zip(results,all_constants,times))
under9 = [triple for triple in aggregate if triple[0]<0.9]

all_con = [i[1] for i in under9]
print(np.mean(all_con),np.std(all_con))

