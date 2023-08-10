# Built and tested on python2
import numpy as np
from tqdm import *
import sys

from scipy import stats

def generate_user_sim_matrix(data_csr, m1_csr, product_matrix):
    # making all unobserved entries in product_matrix as zero
    # makes it simpler for pearson similarity calculation, probably..
    product_matrix = find_and_replace(data=product_matrix, find_value=UNOBSERVED, replace_value=0)
    user_list = np.array(list(set(data_csr[:,0])))
    item_list = np.array(list(set(data_csr[:,1])))

    # Currently using simple pearson similarity:
    user_sim_matrix = np.full((len(user_list), len(user_list)), UNOBSERVED, dtype=float)
    print('Generating user sim matrix (pearson similarity):')
    sys.stdout.flush()
    for user1 in tqdm(user_list):
        for user2 in user_list:
            if user1 >= user2:
                [sim, p_value] = stats.pearsonr(product_matrix[user1], product_matrix[user2])
                if np.isnan(sim):                       # TODO: check if this is valid to do?
                    sim = 0
                user_sim_matrix[user1,user2] = user_sim_matrix[user2,user1] = sim
                # similarity is between -1 and 1
                # therefore, these can be directly used as weights on users' rating for prediction
    return user_sim_matrix





