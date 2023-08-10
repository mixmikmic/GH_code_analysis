import os,sys  
import pandas as pd
import numpy as np
import fasttext

# Set dataset path

data_path = './dbpedia_csv/'

#Load train set
train_file = data_path + 'dbpedia_train.csv'
df = pd.read_csv(train_file, header=None, names=['class','name','description'])

#Load test set
test_file = data_path + 'dbpedia_test.csv'
df_test = pd.read_csv(test_file, header=None, names=['class','name','description'])

#Mapping from class number to class name
class_dict={
1:'Company',
2:'EducationalInstitution',
3:'Artist',
4:'Athlete',
5:'OfficeHolder',
6:'MeanOfTransportation',
7:'Building',
8:'NaturalPlace',
9:'Village',
10:'Animal',
11:'Plant',
12:'Album',
13:'Film',
14:'WrittenWork'
}
df['class_name'] = df['class'].map(class_dict)
df.head()
df.tail()

df.head()


df.tail()

#df.describe().transpose()
desc = df.groupby('class')
desc.describe()

def clean_dataset(dataframe, shuffle=False, encode_ascii=False, clean_strings = False, label_prefix='__label__'):
    # Transform train file
    df = dataframe[['name','description']].apply(lambda x: x.str.replace(',',' '))
    df['class'] = label_prefix + dataframe['class'].astype(str) + ' '
    if clean_strings:
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace('"',''))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace('\'',' \' '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace('.',' . '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace('(',' ( '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace(')',' ) '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace('!',' ! '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace('?',' ? '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace(':',' '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace(';',' '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.lower())
    if shuffle:
        df.sample(frac=1).reset_index(drop=True)
    if encode_ascii :
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii','ignore').str.decode('utf-8'))
    df['name'] = ' ' + df['name'] + ' '
    df['description'] = ' ' + df['description'] + ' '
    return df

get_ipython().run_cell_magic('time', '', "# Transform datasets\ndf_train_clean = clean_dataset(df, True, False)\ndf_test_clean = clean_dataset(df_test, False, False)\n\n# Write files to disk\ntrain_file_clean = data_path + 'dbpedia.train'\ndf_train_clean.to_csv(train_file_clean, header=None, index=False, columns=['class','name','description'] )\n\ntest_file_clean = data_path + 'dbpedia.test'\ndf_test_clean.to_csv(test_file_clean, header=None, index=False, columns=['class','name','description'] )")

df_train_clean.head()

df_train_clean.tail()

df['description'][10]

df_train_clean['description'][10]

get_ipython().run_cell_magic('time', '', "# Train a classifier\noutput_file = data_path + 'dp_model'\nclassifier = fasttext.supervised(train_file_clean, output_file, label_prefix='__label__')")

get_ipython().run_cell_magic('time', '', "# Evaluate classifier\nresult = classifier.test(test_file_clean)\nprint('P@1:', result.precision)\nprint('R@1:', result.recall)\nprint ('Number of examples:', result.nexamples)")



