# Importing Libraries
import os,sys  

# For loading data and doing some exploration
import pandas as pd

# The default import
import numpy as np

# Set path for loading data, saving processed data and saving model
data_path = '~/data/dbpedia_csv/'

# Loading train data
train_file = data_path + 'train.csv'
df = pd.read_csv(train_file, header=None, names=['class','name','description'])

# Loading test data
test_file = data_path + 'test.csv'
df_test = pd.read_csv(test_file, header=None, names=['class','name','description'])

# Data with us
print("Train:{} Test:{}".format(df.shape,df_test.shape))

# Since we have no clue about the classes lets build one
# Mapping from class number to class name
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

# Mapping the classes
df['class_name'] = df['class'].map(class_dict)
df.head()

df.tail()

# What is the group behaviour
desc = df.groupby('class')
desc.describe()

# Lets do some cleaning
def clean_it(text,normalize=True):
    # Replacing possible issues with data. We can add or reduce the replacemtent in this chain
    s = str(text).replace(',',' ').replace('"','').replace('\'',' \' ').replace('.',' . ').replace('(',' ( ').            replace(')',' ) ').replace('!',' ! ').replace('?',' ? ').replace(':',' ').replace(';',' ').lower()
    
    # normalizing / encoding the text
    if normalize:
        s = s.normalize('NFKD').str.encode('ascii','ignore').str.decode('utf-8')
    
    return s

# Now lets define a small function where we can use above cleaning on datasets
def clean_df(data, cleanit= False, shuffleit=False, encodeit=False, label_prefix='__class__'):
    # Defining the new data
    df = data[['name','description']].copy(deep=True)
    df['class'] = label_prefix + data['class'].astype(str) + ' '
    
    # cleaning it
    if cleanit:
        df['name'] = df['name'].apply(lambda x: clean_it(x,encodeit))
        df['description'] = df['description'].apply(lambda x: clean_it(x,encodeit))
    
    # shuffling it
    if shuffleit:
        df.sample(frac=1).reset_index(drop=True)
        
    # for fastext to understand data better
    df['name'] = ' ' + df['name'] + ' '
    df['description'] = ' ' + df['description'] + ' '
        
    return df


get_ipython().run_cell_magic('time', '', '# Transform datasets\ndf_train = clean_df(df, True, True)\ndf_test_cleaned = clean_df(df_test, True, False)')

df_train.head()

df_train.tail()

df['description'][661]

df_train['description'][661]

# Write files to disk
train_file = data_path + 'dbpedia_train.csv'
df_train.to_csv(train_file, header=None, index=False, columns=['class','name','description'] )

test_file = data_path + 'dbpedia_test.csv'
df_test_cleaned.to_csv(test_file, header=None, index=False, columns=['class','name','description'] )

# also small function to see evaluated results.
def print_results(N, p, r):
    print("N\t" + str(N))
    print("Precision {}\t{:.3f}".format(1, p))
    print("Recall    {}\t{:.3f}".format(1, r))

# The library under exploration
import fasttext

from fastText import train_supervised

get_ipython().run_cell_magic('time', '', '# Train a classifier\nmodel = train_supervised(\n    input=train_file, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1\n)\n\n# Evaluating results\nprint_results(*model.test(test_file))\n\n# Saving model\nmodel.save_model(data_path +"basic_model")\n                 ')

get_ipython().run_cell_magic('time', '', '# Classifier retraining\nmodel.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)\n\n# Evaluating\nprint_results(*model.test(test_file))\n\n# again saving retrained model\nmodel.save_model(data_path +"basic_model_quantized")')

