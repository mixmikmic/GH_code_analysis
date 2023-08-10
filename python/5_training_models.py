# Load required libraries
import numpy as np
import pandas as pd
import psycopg2
import nltk
import feature_engineering
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    StratifiedShuffleSplit, RandomizedSearchCV
)

# Set database credentials
db_name1 = 'section1_db'
usernm = 'redwan'
host = 'localhost'
port = '5432'
#pwd = ''

# Prepare a connection to database containing the "About this project" section
con1 = psycopg2.connect(
    database=db_name1, 
    host='localhost',
    user=usernm,
    password=pwd
)

# Query all data from the "About this project" section
sql_query1 = 'SELECT * FROM section1_db;'
section1_df_full = pd.read_sql_query(sql_query1, con1)

# List of meta features to use in models
features = ['num_sents', 'num_words', 'num_all_caps', 'percent_all_caps',
            'num_exclms', 'percent_exclms', 'num_apple_words',
            'percent_apple_words', 'avg_words_per_sent', 'num_paragraphs',
            'avg_sents_per_paragraph', 'avg_words_per_paragraph',
            'num_images', 'num_videos', 'num_youtubes', 'num_gifs',
            'num_hyperlinks', 'num_bolded', 'percent_bolded']

# Select meta features from the dataset
X = section1_df_full[features]

# Remove all rows with no data
X_cleaned = X[~X.isnull().all(axis=1)]

# Fill remaining missing values with zero
X_cleaned = X_cleaned.fillna(0)

# Standardize the meta features
scaler = StandardScaler()
X_std = scaler.fit_transform(X_cleaned)

def preprocess_text(text):
    """Perform text preprocessing such as removing punctuation, lowercasing all
    words, removing stop words and stemming remaining words
    
    Args:
        text (str): cleaned and normalized text of a campaign section
    
    Returns:
        a string containing text that has been preprocessed"""
    
    # Access stop word dictionary
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # Prepare the Porter stemmer
    porter = nltk.PorterStemmer()
    
    # Remove punctuation and lowercase each word
    text = feature_engineering.remove_punc(text).lower()
    
    # Remove stop words and stem each word
    return ' '.join(
        porter.stem(term )
        for term in text.split()
        if term not in stop_words
    )

# Perform preprocessing
#preprocessed_text = section1_df_full.loc[X_cleaned.index, 'normalized_text'] \
#    .apply(preprocess_text)
    
# Alternatively load a pickle that contains the already preprocessed text 
preprocessed_text = joblib.load('data/nlp/preprocessed_text_training_set.pkl')

# Construct a design matrix using an n-gram model
#vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=250)
#X_ngrams = vectorizer.fit_transform(preprocessed_text)

# Alternatively we can load a pickle that contains the already constructed 
# n-grams and a trained vectorizer
X_ngrams = joblib.load('data/nlp/X_ngrams_250.pkl')

vectorizer = joblib.load('data/nlp/vectorizer_250.pkl')

# Convert the meta features into a sparse matrix
X_std_sparse = sparse.csr_matrix(X_std)

# Concatenate the meta features with the n-grams
X_full = sparse.hstack([X_std_sparse, X_ngrams])

# Display the shape of the combined matrix
X_full.shape

# Prepare the classification target variable
y = section1_df_full.loc[X_cleaned.index, 'funded'].to_frame()

# Encode the class labels in the target variable
le = LabelEncoder()
y_enc = le.fit_transform(y.values.ravel())

# Set the recommended number of iterations for stochastic gradient descent
SGD_iterations = np.ceil(10 ** 6 / len(X_std))

# Initialize the hyperparameter space
param_dist = {
    'alpha': np.logspace(-6, -1, 50),
    'l1_ratio': np.linspace(0, 1, 50)
}

# Set up a randomized hyperparameter search and cross-validation strategy
random_search_full = RandomizedSearchCV(
    estimator=SGDClassifier(
        loss='log',
        penalty='elasticnet',
        max_iter=SGD_iterations,
        random_state=41
    ),
    param_distributions=param_dist,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=41),
    scoring='precision',
    random_state=41,
    n_iter=40,
    n_jobs=-1
)

# Train the randomized hyperparameter search to identify optimal 
# hyperparameters
random_search_full.fit(X_full, y_enc)

# Train the classifier on the entire dataset using optimal hyperparameters
clf_full = SGDClassifier(
        loss='log',
        penalty='elasticnet',
        alpha=random_search_full.best_params_['alpha'],
        l1_ratio=random_search_full.best_params_['l1_ratio'],
        max_iter=SGD_iterations,
        random_state=41
)
clf_full.fit(X_full, y_enc);

# Set up a randomized hyperparameter search and cross-validation strategy 
# using meta features only
random_search_meta = RandomizedSearchCV(
    estimator=SGDClassifier(
        loss='log',
        penalty='elasticnet',
        max_iter=SGD_iterations,
        random_state=41
    ),
    param_distributions=param_dist,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=41),
    scoring='precision',
    random_state=41,
    n_iter=40,
    n_jobs=-1
)

# Train a randomized hyperparameter search on the meta features only to 
# identify optimal hyperparameters
random_search_meta.fit(X_std, y_enc)

# Train the classifier on the meta features only using optimal hyperparameters
clf_meta = SGDClassifier(
        loss='log',
        penalty='elasticnet',
        alpha=random_search_meta.best_params_['alpha'],
        l1_ratio=random_search_meta.best_params_['l1_ratio'],
        max_iter=SGD_iterations,
        random_state=41
)
clf_meta.fit(X_std, y_enc);

# Serialize the classifiers, the vectorizer and scaler
#joblib.dump(clf_full, 'trained_classifier.pkl')
#joblib.dump(clf_meta, 'trained_classifier_meta_only.pkl')
#joblib.dump(scaler, 'trained_scaler.pkl')
#joblib.dump(vectorizer, 'vectorizer_250.pkl')

