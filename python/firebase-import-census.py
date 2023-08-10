import pandas as pd
import numpy as np

column_names = [
    'age', 'workclass', 'fnlwgt', 
    'education', 'education-num', 'marital-status', 
    'occupation', 'relationship', 'race', 'sex', 
    'capital-gain', 'capital-loss', 'hours-per-week', 
    'native-country', 'salary']

train_df = pd.read_csv(
    'data/aws/census/adult.data', 
    header=None, names=column_names, 
    sep=', ', engine='python')

test_df = pd.read_csv(
    'data/aws/census/adult.test', 
    header=None, names=column_names, 
    sep=', ', engine='python', skiprows=1)

train_df.shape, test_df.shape

train_df.head()

train_df.to_json(
    orient='index', 
    path_or_buf='data/firebase/census/census.json')
test_df.to_json(
    orient='index', 
    path_or_buf='data/firebase/census/census_test.json')

