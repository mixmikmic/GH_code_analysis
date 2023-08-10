from sklearn.feature_extraction import DictVectorizer

staff = [{'name': 'Steve Miller', 'age': 33.},
         {'name': 'Lyndon Jones', 'age': 12.},
         {'name': 'Baxter Morth', 'age': 18.}]

# Create an object for our dictionary vectorizer
vec =  DictVectorizer()

# Fit then transform the staff dictionary with cev, then output an array
vec.fit_transform(staff).toarray()

# Get Feature Names
vec.get_feature_names()

