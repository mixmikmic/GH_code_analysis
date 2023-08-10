import pandas as pd
import numpy as np

import os

submissions = [file for file in os.listdir('input/') if file[-4 :] == '.csv']
submissions

for file in submissions:
    df = pd.read_csv('input/' + file)
    print(file)
    print(df.head())
    print('-' * 30)

submission_values = []

for file in submissions:
    df = pd.read_csv('input/' + file)
    submission_values.append(df['project_is_approved'].values) # 'project_is_approved is the column we need to predict

submission_values

avg_value = np.mean(submission_values, axis=0)
avg_value

sub_df = pd.DataFrame()
sub_df['id'] = df['id']
sub_df['project_is_approved'] = avg_value
sub_df.head()

sub_df.to_csv('output/avg_sub.csv', index=False, header=True)

scores = [0.79554, 0.7959, 0.80016]
weights = np.array([score / sum(scores) for score in scores])

weights

weighted_value = np.sum(np.array([submission_values[i] * weights[i] for i in range(3)]), axis=0)

weighted_value

sub_df['project_is_approved'] = weighted_value
sub_df.head()

sub_df.to_csv('output/weighted_sub.csv', index=False, header=True)

weights = np.array([0.2, 0.3, 0.5])
weighted_value = np.sum(np.array([submission_values[i] * weights[i] for i in range(3)]), axis=0)

weighted_value

sub_df['project_is_approved'] = weighted_value
sub_df.to_csv('output/weighted_sub_v2.csv', index=False, header=True)

weights = np.array([0.15, 0.3, 0.55])
weighted_value = np.sum(np.array([submission_values[i] * weights[i] for i in range(3)]), axis=0)

weighted_value

sub_df['project_is_approved'] = weighted_value
sub_df.to_csv('output/weighted_sub_v3.csv', index=False, header=True)

