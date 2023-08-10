import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

dataset = pd.read_csv('BaselineDataset.csv')

def create_dataset():
    dataset = np.zeros([600000, 10])
    path = 'Gesture Data/Gesture'
    for i in range(6):
        for j in range(2000):
            
            sample_path = path + str(i+1) + '_Example' + str(j+1) + '.txt'
            sample = pd.read_csv(sample_path, header=None)
            sample['time'] = sample.index
            sample['id'] = 2000*i + j + 1
            sample = np.array(sample)
            startindex = 100000*i + 50*j
            endindex = 50*(j+1) + 100000*i
            #print startindex, endindex
            try:
                dataset[startindex:endindex, :] = sample
            except:
                pass
    return dataset

df = create_dataset()

df = pd.DataFrame(df)
df.head()

df.to_csv('FeatureExtractionDataset.csv')

disp = disp.drop([8,9], axis=1)
disp.columns = ['Sens1', 'Sens2', 'Sens3', 'Sens4', 'Sens5', 'Sens6', 'Sens7', 'Sens8']

disp.describe()

df['Label'] = 'One'
df['Label'][2000:4000] = 'Two'
df['Label'][4000:6000] = 'Three'
df['Label'][6000:8000] = 'Four'
df['Label'][8000:10000] = 'Five'
df['Label'][10000:12000] = 'Six'

from tsfresh import extract_features
extracted_features = extract_features(test, column_id=9, column_sort=8)

extracted_features.shape

extracted_features.head()

impute(extracted_features)

extracted_features

extracted_features.to_csv('ExtractedFeaturesDataset.csv')

extracted_features.head()

sample = pd.read_csv('Gesture Data/Gesture6_Example2000.txt', header=None)

sample[9] = 1.0

from tsfresh.transformers import FeatureAugmenter
augmenter = FeatureAugmenter(column_id=9, column_sort=8)
augmenter.set_timeseries_container(sample)
sample = augmenter.transform(sample)

test_features1 = extract_features(sample, column_id=9, column_sort=8)

