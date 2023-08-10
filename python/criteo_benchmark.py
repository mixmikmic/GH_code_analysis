graphlab_create_product_key = 'YOUR_PRODUCT_KEY'
train_over_subset = True
train_over_transformed = True

running_on_ec2 = True
# running_on_ec2 = False

get_ipython().run_cell_magic('bash', '', '# initialize filesystem on SSD drives\nsudo mkfs -t ext4 /dev/xvdb\nsudo mkfs -t ext4 /dev/xvdc\n\n# create mount points for SSD drives\nsudo mkdir -p /mnt/tmp1\nsudo mkdir -p /mnt/tmp2\n\n# mount SSD drives on created points and temporary file locations\nsudo mount /dev/xvdb /mnt/tmp1\nsudo mount /dev/xvdc /mnt/tmp2\nsudo mount /dev/xvdb /tmp\nsudo mount /dev/xvdc /var/tmp\n\n# set permissions for mounted locations\nsudo chown ubuntu:ubuntu /mnt/tmp1\nsudo chown ubuntu:ubuntu /mnt/tmp2')

get_ipython().run_cell_magic('bash', '', '# Mount EBS data volumn\n# You should attach an EBS volume with at least 500G of space \n# Assuming the disk is mounted at /dev/xvdd\n\nsudo mkdir -p /mnt/data\n\nif grep -qs \'/mnt/data\' /proc/mounts; then\n    echo "EBS volume seems to be already mounted."\nelse\n    sudo mount /dev/xvdd /mnt/data\n    if [ $? -ne 0 ]; then\n        sudo mkfs -t ext4 /dev/xvdd\n        sudo mount /dev/xvdd /mnt/data\n    fi\nfi\n\nsudo chown -R ubuntu:ubuntu /mnt/data')

get_ipython().run_cell_magic('bash', '', 'cd /mnt/data\nfor i in {0..23}; do\n    wget --continue --timestamping http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_${i}.gz\ndone')

import graphlab as gl

if gl.product_key.get_product_key() is None:
    gl.product_key.set_product_key(graphlab_create_product_key)

# Set the cache locations to the SSDs.
if running_on_ec2:
    gl.set_runtime_config("GRAPHLAB_CACHE_FILE_LOCATIONS", "/mnt/tmp1:/mnt/tmp2")

from multiprocessing import cpu_count
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', cpu_count())
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY', 20 * 1024 * 1024 * 1024)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 20 * 1024 * 1024 * 1024)

data_dir = '/mnt/data'

def load_days(start, end):
    data = gl.SFrame()
    for i in range(start, end + 1):
        data = data.append(gl.SFrame.read_csv('%s/day_%d.gz' % (data_dir, i),
                          delimiter='\t', header=False, verbose=False))
    return data

# Load the fit set
fit_set = load_days(0, 19)

# Load the training set
train = load_days(20, 21)

# Create the full training set
full_train = fit_set.append(train)

# Load the testing set
test = load_days(22, 23)

train.print_rows(3)

import time
from datetime import timedelta

if train_over_subset:
    target_feature = 'X1'
    num_features = ['X%d' % (i) for i in xrange(2, 15)] # X2..X14
    cat_features = ['X20', 'X27', 'X31', 'X39']

    start = time.time()

    model = gl.boosted_trees_classifier.create(full_train,
                                               target=target_feature,
                                               validation_set=test,
                                               features=(num_features + cat_features),
                                               max_iterations=5,
                                               random_seed=0)

    print 'End-to-end training time:', timedelta(seconds=(time.time() - start))

# Transform only categorical features
categorical_features = ['X' + str(i) for i in range(15, 41)]

if train_over_transformed:
    start = time.time()

    # Fit the count featurizer on the fit set (first 20 days)
    featurizer = gl.feature_engineering.CountFeaturizer(features=categorical_features, target='X1')
    featurizer.fit(fit_set)

    # Transform the training set (days 21, 22) using the featurizer
    transformed_train = featurizer.transform(train)

    # Transform the testing set (days 23, 24) using the featurizer
    transformed_test = featurizer.transform(test)

    fit_transform_time = time.time() - start
    print 'Fitting the count featurizer and transforming the data time:', timedelta(seconds=fit_transform_time)
    
    # See the transformed data
    transformed_train.print_rows(3)

if train_over_transformed:
    start = time.time()

    model = gl.boosted_trees_classifier.create(transformed_train,
                                               target='X1', 
                                               validation_set=transformed_test,
                                               max_iterations=5,
                                               random_seed=0)

    training_time = time.time() - start
    print 'Training time:', timedelta(seconds=training_time)
    print 'End-to-end fitting and training time', timedelta(seconds=(fit_transform_time + training_time))

