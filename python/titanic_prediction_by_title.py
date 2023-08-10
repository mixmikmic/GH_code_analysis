import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')

data = pd.read_csv('../train.csv')

data['Gender'] = data['Sex'].map({'male':0, 'female':1})

data.head()

import re

data['Title'] = data['Name'].map(lambda name: re.search('[a-zA-Z]+\.',name).group(0))

data['Title'].value_counts()

def classify_title(title):
    if title in ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Rev.', 'Dr.']:
        return title
    elif title == 'Mme.':
        return 'Mrs.'
    elif title == 'Mlle.':
        return 'Miss.'
    else:
        return 'Rare'

data['Title'] = data['Title'].map(classify_title)

data['Title'].value_counts()

y = data['Survived']

df = pd.read_csv('../test.csv')

df.info()

df['Title'] = df['Name'].map(lambda name: re.search('[a-zA-Z]+\.',name).group(0))

df['Title'] = df['Title'].map(classify_title)

df['Title'].value_counts()

df['Survived'] = df['Title'].map({'Mr.':0, 'Miss.':1,'Mrs.':1,'Master.':1,'Rare':1,'Rev.':0,'Dr.':0})

submission = pd.DataFrame({ 'PassengerId': df['PassengerId'], 'Survived': df['Survived']})

# uncomment below to generate the submission csv
# submission.to_csv('submission_prediction_by_title.csv', index=False)

