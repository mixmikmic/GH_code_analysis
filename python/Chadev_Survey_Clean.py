import numpy as np
import pandas as pd
from pandas import Series, DataFrame

pd.set_option('display.max_columns', 200)

# Data cleaning: standardization, missing data done on the cheap outside of the Python environment

survey_df_raw = pd.read_csv(filepath_or_buffer='chadev_survey_cleaned.csv', true_values=['Yes'], false_values=['No'], na_values=['-1'])

survey_df_raw.tail()

# Binning: Categorize company size by number of employees

def companySizeCat(numEmployees):
    if numEmployees < 7: # micro
        return 1
    elif numEmployees < 250: # small
        return 2
    elif numEmployees < 500: # medium
        return 3
    elif numEmployees < 1000: # large
        return 4
    else: # enterprise
        return 5 

survey_df_raw['COMP_SIZE'] = survey_df_raw['COMP_SIZE'].apply(lambda x: companySizeCat(x))

survey_df_raw.head()

# Drop columns that don't give any particular insight (subjectively chosen)

survey_df_raw.drop(['Timestamp', 'DES_NUM_HOURS_WORK', 'HEALTH_INS_PCT', 'PAID_LUNCH_HOUR', 
                    'PUBLIC_SPEAKING', 'FREE_TESLA', 'RIGHT_LEFT_BRAIN', 'HIGH_SCHOOL'], axis=1, inplace=True);

survey_df_raw.columns

# Let's try to understand the data

survey_df_raw.describe()

sel_columns_df = survey_df_raw[['COMP_SIZE','YEARS_EXP', 'WORK_HIST_COUNT', 'BOOKS_READ', 'SALARY']]
sel_columns_df.mode()

# Define a function to tokenize the raw text entered by the users 
def tokenizr(text):
    tokens = text.split(";");
    return [ token.strip() for token in tokens if token.isspace() == False ]
    

# Vectorize the text using sklearn provided CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tools = survey_df_raw['STACK'];
tools.fillna('', inplace=True);

tf_vectorizer = CountVectorizer(tokenizer=tokenizr);
tools_matrix = tf_vectorizer.fit_transform(tools);

# Produces a matrix
tools_matrix.shape

tf_vectorizer.get_feature_names()

# Turn the matrix into a dataframe
tools_matrix_df = DataFrame(data=tools_matrix.toarray(), columns=tf_vectorizer.get_feature_names())

# Join the dataframe to the survey data frame
survey_df_wTM = survey_df_raw.join(tools_matrix_df)
survey_df_wTM.drop(['STACK'], axis=1, inplace=True)

# visualize
survey_df_wTM.head()

# Get the most mentioned stack tools by summing the values of the tools_matrix_df
# Results show we have a very webby crowd (or at the very least, lots of full-stack developers)
X = tools_matrix_df.sum()

X.sort(ascending=False)
X.head(10)

# Using a basic KMeans algorithm
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, max_iter=500)

# Attempt 1: Just give it all the data in its raw, naked glory and see what it comes up with
kmeans.fit(survey_df_wTM)
kmeans.labels_

# Taking a look at what records were assigned to what cluster, I notice a disturbing trend: Salary is outweighing 
#   all other features
survey_df_c1 = survey_df_wTM.copy()
survey_df_c1.insert(0, 'CLUSTER1_LABEL', kmeans.labels_)
survey_df_c1.head(10)

survey_df_c1.tail(10)

# To prove that out a bit more I group by the cluster label and run some stats
cl1_groups = survey_df_c1.groupby(['CLUSTER1_LABEL'])
cl1_groups.agg([np.mean, np.min, np.max, np.std])

# But what if I wanted to stop salary from outweighing the other features so that the algorithm can consider ALL
#   features in its clustering decision?

def normalize(df, columns):
    result = df.copy()
    for feature_name in columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

# apply it to...
survey_df_normalized = normalize(survey_df_raw, ['YEARS_EXP', 'NUM_HOURS_WORK', 'DAYS_OFF', 'SALARY', 'CONF_BUDGET', 'BOOKS_READ', 'WORK_HIST_COUNT'])
survey_df_normalized.drop(['STACK', 'COMP_MEALS'], axis=1, inplace=True)
survey_df_normalized.head(10)

kmeans.fit(survey_df_normalized)

survey_df_c2 = survey_df_wTM.copy()
survey_df_c2.insert(0, 'CLUSTER2_LABEL', kmeans.labels_)
survey_df_c2.head(10)

# Fixed! But there is a new problem. I don't know what these clusters mean because there are too many dimensions
#   for me to wrap my tiny little brain around.

cl2_groups = survey_df_c2.groupby(['CLUSTER2_LABEL'])
cl2_groups.agg([np.mean, np.min, np.max, np.std])

# Since I'm looking for clusters of ... say devs that earn similar salaries and have similar numbers of paid holidays and 
#    have similar work histories, why don't I simply cluster on those 3 dimensions?

survey_df_selected = survey_df_normalized[['SALARY', 'DAYS_OFF', 'WORK_HIST_COUNT']]

kmeans = KMeans(n_clusters=5, max_iter=500)
kmeans.fit(survey_df_selected)
kmeans.labels_

survey_df_c3 = survey_df_raw.copy()[['SALARY', 'DAYS_OFF', 'WORK_HIST_COUNT']]
survey_df_c3.insert(0, 'CLUSTER3_LABEL', kmeans.labels_)
survey_df_c3.head(10)

cl3_groups = survey_df_c3.groupby(['CLUSTER3_LABEL'])
cl3_groups.agg([np.mean, np.min, np.max, np.std])

get_ipython().magic('matplotlib inline')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

survey_df_c3b = survey_df_normalized.copy()[['SALARY', 'DAYS_OFF', 'WORK_HIST_COUNT']]
survey_df_c3b.insert(0, 'CLUSTER3_LABEL', kmeans.labels_)
cl3b_groups = survey_df_c3b.groupby(['CLUSTER3_LABEL'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(cl3b_groups.get_group(0)['SALARY'].tolist(), 
           cl3b_groups.get_group(0)['DAYS_OFF'].tolist(), 
           cl3b_groups.get_group(0)['WORK_HIST_COUNT'].tolist(), 
           c='r', marker='o')
ax.scatter(cl3b_groups.get_group(1)['SALARY'].tolist(), 
           cl3b_groups.get_group(1)['DAYS_OFF'].tolist(), 
           cl3b_groups.get_group(1)['WORK_HIST_COUNT'].tolist(), 
           c='b', marker='o')
ax.scatter(cl3b_groups.get_group(2)['SALARY'].tolist(), 
           cl3b_groups.get_group(2)['DAYS_OFF'].tolist(), 
           cl3b_groups.get_group(2)['WORK_HIST_COUNT'].tolist(), 
           c='y', marker='^')
ax.scatter(cl3b_groups.get_group(3)['SALARY'].tolist(), 
           cl3b_groups.get_group(3)['DAYS_OFF'].tolist(), 
           cl3b_groups.get_group(3)['WORK_HIST_COUNT'].tolist(), 
           c='g', marker='x')
ax.scatter(cl3b_groups.get_group(4)['SALARY'].tolist(), 
           cl3b_groups.get_group(4)['DAYS_OFF'].tolist(), 
           cl3b_groups.get_group(4)['WORK_HIST_COUNT'].tolist(), 
           c='w', marker='o')

ax.set_xlabel('SALARY')
ax.set_ylabel('DAYS_OFF')
ax.set_zlabel('WORK_HIST_COUNT')

plt.show()

from sklearn.tree import DecisionTreeClassifier, export_graphviz

y = survey_df_raw["SALARY"]
X = survey_df_raw.drop(["SALARY", "STACK"], axis=1, inplace=False)
tree = DecisionTreeClassifier(min_samples_split=4)
tree.fit(X, y)

from os import system
from IPython.display import Image

with open('tree.dot', 'w') as dotfile:
    export_graphviz(tree, dotfile, feature_names=X.columns)

system("dot -Tpng tree.dot -o tree.png")
Image(filename='tree.png')



