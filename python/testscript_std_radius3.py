from datetime import datetime
datetime.now().time()     # (hour, min, sec, microsec)

# if this way of importing another jupyter notebook fails for you
# then you can use any one of the many methods described here:
# https://stackoverflow.com/questions/20186344/ipynb-import-another-ipynb-file
get_ipython().run_line_magic('run', "'../src/finalcode.ipynb'")

datetime.now().time()     # (hour, min, sec, microsec)

'''Dataset Parameters'''
################################################################################################################
DATA_PATH = '../data/ml-100k/u.data' # ml-100k data set has 100k ratings, 943 users and 1682 items
DELIMITER = "\t"               # tab separated or comma separated data format
N_RATINGS = 100000
################################################################################################################

# These parameters will be detected automatically from dataset
# -1 is for the default value
FIRST_INDEX = -1
USERS = -1
ITEMS = -1
SPARSITY = -1                  # 'p' in the equations
UNOBSERVED = 0                 # default value in matrix for unobserved ratings; prefer to keep it 0

# To reduce size of csr for testing purpose
# WARNING: ONLY TO BE USED FOR TESTING
# (for real run, put SIZE_REDUCTION = False)
SIZE_REDUCTION = False
#USER_LIMIT = 50
#ITEM_LIMIT = 100

'''Hyperparameters'''
# All the hyperparameters have default values
#To use them, set the parameters as -1
################################################################################################################
TRAIN_TEST_SPLIT = -1                   # %age of test ratings wrt train rating ; value in between 0 and 1
C1 = -1                                 # probability of edges in training set going to E1
C2 = -1                                 # probability of edges in training set going to E2
RADIUS = 3                              # radius of neighborhood, radius = # edges between start and end vertex
UNPRED_RATING = 3                       # rating (normalized) for which we dont have predicted rating between 1 - 5
THRESHOLD = 0.01                        # distance similarity threshold used for rating prediction
################################################################################################################

# checks on hyper parameters    
if isinstance(C1, float) and isinstance(C2, float) and (C1 > 0) and (C2 > 0) and 1 - C1 - C2 > 0:
    print('c1 = {}'.format(C1))
    print('c2 = {}'.format(C2))
    print('c3 = {}'.format(1-C1-C2))
elif (C1 == -1) and (C2 == -1):
    C1 = C2 = 0.33
    print('c1 = {} (default)'.format(C1))
    print('c2 = {} (default)'.format(C2))
    print('c3 = {} (default)'.format(1-C1-C2))
else:
    print('ERROR: Incorrect values set for C1 and C2')
    
if isinstance(RADIUS, int) and RADIUS > 0:
    print('Radius = {}'.format(RADIUS))
elif RADIUS == -1:
    print('Radius = default value as per paper')
else:
    print('ERROR: Incorrect values set for Radius')

if UNPRED_RATING >= 1 and UNPRED_RATING <= 5:
    print('Rating set for unpredicted ratings = {}'. format(UNPRED_RATING))
elif UNPRED_RATING == -1:
    UNPRED_RATING = 3
    print('Rating set for unpredicted ratings = {} (default)'. format(UNPRED_RATING))
else:
    print('ERROR: Incorrect values set for UNPRED_RATING')
    
if TRAIN_TEST_SPLIT > 0 and TRAIN_TEST_SPLIT < 1:
    print('TRAIN_TEST_SPLIT = {}'.format(TRAIN_TEST_SPLIT))
elif TRAIN_TEST_SPLIT == -1:
    TRAIN_TEST_SPLIT = 0.2
    print('TRAIN_TEST_SPLIT = 0.2 (default)')
else:
    print('ERROR: Incorrect values set for TRAIN_TEST_SPLIT')



data_csr = read_data_csr(fname=DATA_PATH, delimiter=DELIMITER)

if SIZE_REDUCTION:
    data_csr = reduce_size_of_data_csr(data_csr)

if data_csr.shape[0] == N_RATINGS:  # gives total no of ratings read; useful for verification
    print('Reading dataset: done')
else:
    print('Reading dataset: FAILED')
    print( '# of missing ratings: ' + str(N_RATINGS - data_csr.shape[0]))
    
check_and_set_data_csr(data_csr=data_csr)

[train_data_csr, test_data_csr] = generate_train_test_split_csr(data_csr=data_csr, split=TRAIN_TEST_SPLIT)

train_data_csr = normalize_ratings_csr(train_data_csr)
train_data_csr = csr_to_symmetric_csr(train_data_csr)
# the symmetric matrix obtained doesnt contain repititions for any user item pair
# only the item_ids are scaled by item_ids += USERS
# hence, we can safely go ahead and use this CSR matrix for sample splitting step



[m1_csr, m2_csr, m3_csr] = sample_splitting_csr(data_csr=train_data_csr, c1=C1, c2=C2)



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



distance_matrix = compute_distance_matrix(r_neighbor_matrix, r1_neighbor_matrix, m2_csr)

describe_distance_matrix(distance_matrix)



# prefer to choose a threshold now based on describe_distance_matrix
THRESHOLD = 2

sim_matrix = generate_sim_matrix(distance_matrix, threshold=THRESHOLD)

# Prepare the test dataset using Model preparation section functions
test_data_csr = normalize_ratings_csr(test_data_csr)
test_data_csr = csr_to_symmetric_csr(test_data_csr)

# Getting estimates for only test data points
prediction_array = generate_averaged_prediction_array(sim_matrix, m3_csr, test_data_csr)

# To generate complete rating matrix do the following:
#prediction_matrix = generate_averaged_prediction_matrix(sim_matrix, m3_csr)



# We have already prepared the test data (required for our algorithm)
y_actual  = test_data_csr[:,2]
y_predict = prediction_array
# If we want, we could scale our ratings back to 1 - 5 range for evaluation purposes
#But then paper makes no guarantees about scaled ratings
#y_actual  = y_actual * 5
#y_predict = y_predict * 5

get_rmse(y_actual, y_predict)

get_avg_err(y_actual, y_predict)

check_mse(m1_csr, y_actual, y_predict)



datetime.now().time()     # (hour, min, sec, microsec)



