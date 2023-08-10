from os import chdir
chdir('/home/jovyan/work')

import random
import pandas as pd

get_ipython().system('ls -l')

get_ipython().system('mkdir data')

get_ipython().system('wget -P data/   http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')

get_ipython().system('wc -l data/adult.data')

get_ipython().system('head -n 2 data/adult.data')

number_of_rows = 32562
sample_size = 3300

rows_to_skip = random.sample(range(number_of_rows), number_of_rows - sample_size)
rows_to_skip.sort()

adult_df = pd.read_csv('data/adult.data', header=None, skiprows=rows_to_skip)
adult_df.columns = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'income_label'
]

adult_df.sample(3)

adult_df.dtypes



