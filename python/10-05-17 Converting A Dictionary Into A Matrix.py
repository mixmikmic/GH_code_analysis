# Load library
from sklearn.feature_extraction import DictVectorizer

# Our dictionary of data
data_dict = [{'Red': 2, 'Blue': 4},
             {'Red': 4, 'Blue': 3},
             {'Red': 1, 'Yellow': 2},
             {'Red': 2, 'Yellow': 2}]

# Create DictVectorizer object
dictvectorizer = DictVectorizer(sparse=False)

# Convert dictionary into feature matrix
features = dictvectorizer.fit_transform(data_dict)

# View feature matrix
features

# View feature matrix column names
dictvectorizer.get_feature_names()

