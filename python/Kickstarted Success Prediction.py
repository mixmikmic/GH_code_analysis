import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.rcParams['figure.figsize']=6,6

df = pd.read_csv('dataset/kickstarter/train.csv')

df.head()

df['all_text'] = df['name'] + " "+ df['desc'] +" "+ df['keywords'].apply(lambda x: x.replace('-', ' '))

### Distribution of kickstarter projects by country

sns.countplot(x='country', data= df)

aggregated = df.groupby(['country','final_status']).size().reset_index().rename(columns={0: 'proj_counts'})

by_projects = aggregated.groupby(['country','final_status']).agg({'proj_counts': 'sum'})

pct_projects = by_projects.groupby(level=0).apply(lambda x: x / float(x.sum())).reset_index().rename(columns={'proj_counts': 'final_status_pct'})

sns.factorplot(x='country', y='final_status_pct', hue='final_status', data=pct_projects, kind='bar', size=6)

sns.countplot(x='goal', data=df[df['goal'] < 50000], order=df.goal.value_counts().iloc[:10].index)

df.columns

df['duration'] = ((df['deadline'] - df['launched_at'])/86400.0).astype(int)

df['days_status_changed'] = ((df['deadline'] - df['state_changed_at'])/86400.0).astype(int)

df['country'] = pd.Categorical(df.country)
df['currency'] = pd.Categorical(df.currency)
df['country_int'] = df.country.cat.codes
df['currency_int'] = df.currency.cat.codes



from nltk.corpus import stopwords
cachedStopWords = stopwords.words('english')

# Function for removing stop words from a string
def removeStopwords(s):
    return ' '.join([word for word in s.split() if word not in cachedStopWords])

# Function for cleaning the reviews
def cleanText(s):
    s = str(s).lower()                         # Convert to lowercase
    s = s.replace(r'<.*?>', ' ')          # Remove HTML characters
    s = s.replace('"', '')               # Remove single quotes ' 
    s = s.replace('\'', '')               # Remove single quotes ' 
    s = s.replace('-', '')                # Remove dashes -
    s = s.replace(r'[^a-zA-Z]', ' ')      # Remove non alpha characters
    s = s.strip()                         # Remove whitespace at start and end
    s = re.sub(r'[^\w\s]','',s)
    return s

import string
import re

df['cleaned_text'] = df['all_text'].apply(lambda x: cleanText(x))

df[['cleaned_text', 'goal', 'duration', 'final_status']].head()

features = ['goal', 'disable_communication', 'country_int', 'currency_int', 'duration', 'days_status_changed', 'backers_count', 'cleaned_text']

int_features = ['goal', 'disable_communication', 'country_int', 'currency_int', 'duration', 'days_status_changed', 'backers_count']

from sklearn.cross_validation import train_test_split
X = df[features]
y = df['final_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


from sklearn.base import BaseEstimator, TransformerMixin

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, keys):
        self.keys = keys

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return [x[0] for x in data_dict[self.keys].values.tolist()]

class IntItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, keys):
        self.keys = keys

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.keys].astype(float).values
    
t = ItemSelector(['cleaned_text'])



from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            ('pipeline', Pipeline([
                ('selector', ItemSelector(['cleaned_text'])),
                ('vect', TfidfVectorizer(stop_words='english', min_df=5, max_df=50))
            ])),
            
            #Pipeline for pulling ad hoc features from post's body
            ('integer_features', Pipeline([('fts', IntItemSelector(int_features))])),
        ]
    )),

    # Use a SVC classifier on the combined features
#     ('svc', SVC(random_state=12, kernel='linear', probability=True, class_weight='balanced')),
#      ('clf', LogisticRegression())
      ('clf', RandomForestClassifier(n_estimators=100))
])

pipeline.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix
predicted = pipeline.predict(X_test)
print(np.mean(predicted == y_test))
print(confusion_matrix(y_test, predicted))
from sklearn import metrics
print(metrics.classification_report(y_test, predicted))

parameters = {'union__pipeline__vect__ngram_range': [(1, 1), (1, 2), (1,3)],
              'union__pipeline__vect__use_idf': (True, False),
              'clf__C': (1e-2, 1e-3, 0.1, 1, 10),
}

gs_clf = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
gs_clf = gs_clf.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
predicted = gs_clf.predict(X_test)
print(np.mean(predicted == y_test))
print(confusion_matrix(y_test, predicted))
from sklearn import metrics
print(metrics.classification_report(y_test, predicted))

df.final_status.value_counts()

(73568)/(73568+34561)







