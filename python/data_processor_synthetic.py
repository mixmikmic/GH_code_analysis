get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import numpy as np
import pickle 

import sys
sys.path.append('../src')
import data_utils

#load the data
data_path = '/data2/data/zebrahim/synthetic_dataset/data_8192_1000_3_4_4_.03_.03_.2/data_8192_1000_3_4_4_.03_.03_.2.mat'
data_raw = data_utils.load_mat(data_path, 'data')

#reshape data to transform 4 variables of 3 modalities of a time series to 12 variables.
data_reshaped = np.reshape(data_raw, (data_raw.shape[0], -1, data_raw.shape[-1]))
#reshape to have (batch_size, signal_len, num_vari)
data_transposed = np.transpose(data_reshaped, (0, 2, 1))

#parameters
window_size = 8192
stride = 8192

data_raw_train = data_transposed[:890]
data_raw_validation= data_transposed[890: 900]
data_raw_test = data_transposed[900:1000]

_, data_stacked_train = data_utils.slide_window(data_raw_train,
                                                    window_size,
                                                    stride,
                                                    num_dim_expand=0)

_, data_stacked_validation = data_utils.slide_window(data_raw_validation,
                                               window_size,
                                               stride,
                                               num_dim_expand=0)


_, data_stacked_test = data_utils.slide_window(data_raw_test,
                                               window_size,
                                               stride,
                                               num_dim_expand=0)

gt_raw = data_utils.load_mat(data_path, 'gtruth')
gt_raw_train = gt_raw[:890]
gt_raw_validation = gt_raw[890:900]
gt_raw_test = gt_raw[900:1000]

_, gt_stacked_train = data_utils.slide_window(gt_raw_train,
                                              window_size,
                                              stride,
                                              num_dim_expand=0)

_, gt_stacked_validation = data_utils.slide_window(gt_raw_validation,
                                              window_size,
                                              stride,
                                              num_dim_expand=0)

_, gt_stacked_test = data_utils.slide_window(gt_raw_test,
                                             window_size,
                                             stride,
                                             num_dim_expand=0)

#index of changes
gt_mean = data_utils.load_mat(data_path, 'gt_mean')
gt_mean_test = gt_mean[900:1000]

data = {}
data['train_data'] = data_stacked_train
data['train_gt'] = gt_stacked_train

data['validation_data'] = data_stacked_validation
data['validation_gt'] = gt_stacked_validation

data['test_data'] = data_stacked_test
data['test_gt'] = gt_stacked_test

data['index_of_changes'] = gt_mean_test

path_to_save = '/data2/data/zebrahim/synthetic_dataset/data_8192_1000_3_4_4_.03_.03_.2'
with open(path_to_save+'processed_data2.p', 'w') as fout:
    pickle.dump(data, fout)

1









