# if this way of importing another jupyter notebook fails for you
# then you can use any one of the many methods described here:
# https://stackoverflow.com/questions/20186344/ipynb-import-another-ipynb-file
get_ipython().run_line_magic('run', "'revisedcode.ipynb'")





'''Dataset Parameters'''
DATA_PATH = './ml-100k/u.data' # ml-100k data set has 100k ratings, 943 users and 1682 items
#DATA_TYPE =  0              # 0: CSR format, 1: 2D matrix format  # TODO: use it
DELIMITER = "\t"             # tab separated or comma separated data format
FIRST_INDEX = 1
N_RATINGS = 100000
USERS = 943
ITEMS = 1682

'''Hyperparameters'''
C1 = 0.2                # probability of edges in training set going to E1
C2 = 0.3                # probability of edges in training set going to E2
C3 = 1 - C1 - C2        # probability of edges in training set going to E3
RADIUS = 3              # radius of neighborhood, radius = # edges between start and end vertex, keep it -1 to use default value given in paper
THRESHOLD = 943

#checks on parameters
if C3 <= 0:
    print('ERROR: Please set the values of C1 and C2, s.t, C1+C2 < 1')

'''Hardcoding values'''
OFFSET = USERS + 10                     # offset so that user_id and item_id are different in graph; keep it >= #USERS
UNOBSERVED = -1
GET_PRODUCT_FAIL_RETURN = UNOBSERVED    #TODO: This hardcoding can be removed in future
TRAIN_TEST_SPLIT = 0.2                  # %age of test ratings wrt train rating ; value in between 0 and 1
AVG_RATING = 3                          # ratings for which we dont have predicted rating

data_csr = read_data_csr(fname=DATA_PATH, delimiter=DELIMITER)

if data_csr.shape[0] == N_RATINGS:  # gives total no of ratings read; useful for verification
    print('Reading dataset: done')
else:
    print('Reading dataset: FAILED')
    #print( '# of missing ratings: ' + str(N_RATINGS - data_csr.shape[0]))  #TODO

check_dataset_csr(data_csr=data_csr)

#TODO : normalize the ratings and symmtericize the given matrix

[train_data_csr, test_data_csr] = generate_train_test_split_csr(data_csr=data_csr, split=TRAIN_TEST_SPLIT)



[m1_csr, m2_csr, m3_csr] = sample_splitting_csr(data_csr=data_csr, c1=C1, c2=C2, shuffle=False)



product_matrix = generate_product_matrix(data_csr, m1_csr, c1=C1, radius=RADIUS)
#TODO: check why generating product matrix is taking about a minute longer w.r.t. rawcode



user_sim_matrix = generate_user_sim_matrix(data_csr, m1_csr, product_matrix)
# del product_matrix



predicted_matrix = generated_weighted_averaged_prediction_matrix(data_csr, m3_csr, user_sim_matrix, bounded=True)
# del user_sim_matrix



[y_actual, y_predict] = generate_true_and_test_labels(test_data_csr, predicted_matrix)
# del predicted_matrix

get_rmse(y_actual, y_predict)

get_avg_err(y_actual, y_predict)

check_mse(data_csr, y_actual, y_predict) # TODO: this might be because the matrix considered here is not symmetric?





