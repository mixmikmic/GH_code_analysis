import os
import sys
import cPickle as pickle

module_path = os.path.abspath(os.path.join('../code'))
if module_path not in sys.path:
    sys.path.append(module_path)

from medoids import investigate_stability, make_distance_array

#getting distances
with open('../data/distances.pkl', 'r') as f:
    distance_dictionary = pickle.load(f)
    
movies, distances = make_distance_array(distance_dictionary)

d_inter = investigate_stability(movies, distances, 10, 3)

with open('../data/3clusters.pkl', 'w') as f:
    pickle.dump(d_inter, f)

d_inter[0]

d_inter[1]

d_inter[2]

