from datetime import datetime
datetime.now().time()     # (hour, min, sec, microsec)

# if this way of importing another jupyter notebook fails for you
# then you can use any one of the many methods described here:
# https://stackoverflow.com/questions/20186344/ipynb-import-another-ipynb-file
get_ipython().run_line_magic('run', "'../src/finalcode.ipynb'")

datetime.now().time()     # (hour, min, sec, microsec)

FIRST_INDEX = -1
USERS = -1
ITEMS = -1
SPARSITY = -1                  # 'p' in the equations
UNOBSERVED = 0                 # default value in matrix for unobserved ratings
N_RATINGS = 7
C1 = 0                         # only to account for scale_factor in step 3
C2 = 1                         # only to account for scale_factor in step 3

RADIUS = 3                              # radius of neighborhood, radius = # edges between start and end vertex
UNPRED_RATING = -1                      # rating (normalized) for which we dont have predicted rating



m1_csr = read_data_csr(fname='../data/very_small_graph.txt', delimiter="\t")
check_and_set_data_csr(data_csr=m1_csr)

m1_csr = normalize_ratings_csr(m1_csr)          ##### REMOVE THIS CELL
m1_csr = csr_to_symmetric_csr(m1_csr)



# This step is  being skipped (not needed) for very_small_graph.txt dataset



[r_neighbor_matrix, r1_neighbor_matrix] = generate_neighbor_boundary_matrix(m1_csr)
# all neighbor boundary vector for each user u is stored as u'th row in neighbor_matrix
# though here the vector is stored a row vector, we will treat it as column vector in Step 4
# Note: we might expect neighbor matrix to be symmetric with dimensions (USERS+ITEMS)*(USERS+ITEMS)
#     : since distance user-item and item-user should be same
#     : but this is not the case since there might be multiple paths between user-item
#     : and the random path picked for user-item and item-user may not be same
#     : normalizing the matrix also will result to rise of difference

describe_neighbor_count(r_neighbor_matrix)

describe_neighbor_count(r1_neighbor_matrix)



distance_matrix = compute_distance_matrix(r_neighbor_matrix, r1_neighbor_matrix, m1_csr)
distance_matrix

describe_distance_matrix(distance_matrix)



sim_matrix = generate_sim_matrix(distance_matrix, threshold=.26)
sim_matrix

prediction_array = generate_averaged_prediction_array(sim_matrix, m1_csr, m1_csr)
prediction_array

prediction_matrix = generate_averaged_prediction_matrix(sim_matrix, m1_csr)
prediction_matrix



# We have already prepared the test data (required for our algorithm)
test_data_csr = m1_csr
y_actual  = test_data_csr[:,2]
y_predict = prediction_array
# If we want, we could scale our ratings back to 1 - 5 range for evaluation purposes
#But then paper makes no guarantees about scaled ratings
#y_actual  = y_actual * 5
#y_predict = y_actual * 5

get_rmse(y_actual, y_predict)

get_avg_err(y_actual, y_predict)

check_mse(m1_csr, y_actual, y_predict)



datetime.now().time()     # (hour, min, sec, microsec)



