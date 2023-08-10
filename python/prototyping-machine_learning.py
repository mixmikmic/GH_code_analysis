# Load required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import nltk
import feature_engineering
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    learning_curve, StratifiedShuffleSplit, cross_val_score, ShuffleSplit,
    cross_val_predict, RandomizedSearchCV
)

# Set figure display options
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(context='notebook', style='darkgrid')
sns.set(font_scale=1.3)

# Set database credentials
db_name1 = 'section1_db'
db_name2 = 'section2_db'
usernm = 'redwan'
host = 'localhost'
port = '5432'
#pwd = ''

# Connect to database containing the "About this project" section
con1 = psycopg2.connect(
    database=db_name1, 
    host='localhost',
    user=usernm,
    password=pwd
)

# Connect to database containing the "Risks and challenges" section
con2 = psycopg2.connect(
    database=db_name2, 
    host='localhost',
    user=usernm,
    password=pwd
)

# Query all data from both campaign sections
sql_query1 = 'SELECT * FROM section1_db;'
sql_query2 = 'SELECT * FROM section2_db;'
section1_df_full = pd.read_sql_query(sql_query1, con1)
section2_df_full = pd.read_sql_query(sql_query2, con2)

# Define a target variable for regression
section1_df_full['percent_funded'] = section1_df_full['pledged'] /     section1_df_full['goal']

# Display a few rows
section1_df_full.head(2)

# List of meta features to use in models
features = ['num_sents', 'num_words', 'num_all_caps', 'percent_all_caps',
            'num_exclms', 'percent_exclms', 'num_apple_words',
            'percent_apple_words', 'avg_words_per_sent', 'num_paragraphs',
            'avg_sents_per_paragraph', 'avg_words_per_paragraph',
            'num_images', 'num_videos', 'num_youtubes', 'num_gifs',
            'num_hyperlinks', 'num_bolded', 'percent_bolded']

# Select meta features from the dataset
X = section1_df_full[features]

# Display the first five rows of the design matrix
X.head()

# Remove all rows with no data
X_cleaned = X[~X.isnull().all(axis=1)]

# Fill remaining missing values with zero
X_cleaned = X_cleaned.fillna(0)

# Identify projects missing a "Risks and challenges" section
section2_df_full['risks_present'] = ~section2_df_full['normalized_text']     .isnull()

# Select the projects missing a "Risks and challenges" section from the table
# containing the "About this projects" section
section1_df_full = section1_df_full.merge(
    section2_df_full[['index', 'risks_present']],
    how='left', 
    on='index'
)

# Compute the proportion of projects missing a "Risks and challenges" section
(len(section1_df_full) - section1_df_full['risks_present'].sum()) /     len(section1_df_full)

# Compute the proportion of projects missing an "About this projects" section
(len(section1_df_full) - len(X_cleaned)) / len(section1_df_full)

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

# Construct a design matrix using an n-gram model and tf-idf statistics
#vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=250)
#X_ngrams = vectorizer.fit_transform(preprocessed_text)
#joblib.dump(vectorizer, 'vectorizer_250.pkl')
#joblib.dump(X_ngrams, 'X_ngrams_250.pkl')

# Alternatively we can load a pickle that contains the already constructed 
# n-grams and vectorizer
X_ngrams = joblib.load('data/nlp/X_ngrams_250.pkl')
vectorizer = joblib.load('data/nlp/vectorizer_250.pkl')

# Convert the meta features into a sparse matrix
X_std_sparse = sparse.csr_matrix(X_std)

# Concatenate the meta features with the n-grams
X_full = sparse.hstack([X_std_sparse, X_ngrams])

# Display the shape of the combined matrix for confirmation
X_full.shape

# Prepare the regression target variable
y_reg = section1_df_full.loc[X_cleaned.index, 'percent_funded'].to_frame()

# Display a histogram of the regression target variable
sns.distplot(y_reg['percent_funded'], kde=False);

# Display a kde plot of the regression target variable
sns.distplot(
    y_reg[y_reg['percent_funded'] < 5000]['percent_funded'],
    kde=False
);

# Display a kde plot of the regression target variable
sns.distplot(
    y_reg[y_reg['percent_funded'] < 10]['percent_funded'],
    kde=False
);

# Prepare the classification target variable
y = section1_df_full.loc[X_cleaned.index, 'funded'].to_frame()

# Display the class distribution
y['funded'].value_counts()

# Encode the class labels in the target variable
le = LabelEncoder()
y_enc = le.fit_transform(y.values.ravel())

# Set the recommended number of iterations for SGD
SGD_iterations = np.ceil(10 ** 6 / len(X_std))
SGD_iterations

# Compute cross-validated precision scores for an OLS regression
scores = cross_val_score(
    estimator=SGDRegressor(max_iter=SGD_iterations, random_state=41),
    X=X_std,
    y=y_reg.values.ravel(),
    cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=41),
    scoring='r2',
    n_jobs=-1
)

# Display the average and standard deviation of the cross-validation scores
print('R^2: {} +/- {}'.format(scores.mean(), scores.std()))

# Compute cross-validated precision scores for logistic regression
scores = cross_val_score(
    estimator=SGDClassifier(
        loss='log',
        penalty='elasticnet',
        max_iter=SGD_iterations,
        random_state=41
    ),
    X=X_std,
    y=y_enc,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=41),
    scoring='precision',
    n_jobs=-1
)

# Display the average and standard deviation of the cross-validation scores
print('Precision: {} +/- {}'.format(scores.mean(), scores.std()))

# Make cross-validated predictions for the training set
y_pred = cross_val_predict(
    estimator=SGDClassifier(
        loss='log',
        penalty='elasticnet',
        max_iter=SGD_iterations,
        random_state=41
    ),
    X=X_std,
    y=y_enc,
    n_jobs=-1
)

# Compute the confusion matrix
cm = metrics.confusion_matrix(y_enc, y_pred)

# Display a normalized confusion matrix
pd.DataFrame(
    np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2),
    index=[['actual', 'actual'], ['unfunded', 'funded']],
    columns=[['predicted', 'predicted'], ['unfunded', 'funded']]
)

# Compute cross-validated precision scores for logistic regression
scores = cross_val_score(
    estimator=SGDClassifier(
        loss='log',
        penalty='elasticnet',
        max_iter=SGD_iterations,
        random_state=41
    ),
    X=X_full,
    y=y_enc,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=41),
    scoring='precision',
    n_jobs=-1
)

# Display the average and standard deviation of the cross-validation scores
print('Precision: {} +/- {}'.format(scores.mean(), scores.std()))

# Make cross-validated predictions for the training set
y_pred = cross_val_predict(
    estimator=SGDClassifier(
        loss='log',
        penalty='elasticnet',
        max_iter=SGD_iterations,
        random_state=41
    ),
    X=X_full,
    y=y_enc,
    n_jobs=-1
)

# Compute the confusion matrix
cm2 = metrics.confusion_matrix(y_enc, y_pred)

# Display a normalized confusion matrix
pd.DataFrame(
    np.round(cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis], 2),
    index=[['actual', 'actual'], ['unfunded', 'funded']],
    columns=[['predicted', 'predicted'], ['unfunded', 'funded']]
)

# Select 10 different sizes of the complete dataset
sample_space = np.linspace(100, len(X_std) * 0.8, 10, dtype='int')

# Compute learning curves
train_sizes, train_scores, valid_scores = learning_curve(
    estimator=SGDClassifier(
        loss='log',
        penalty='elasticnet',
        max_iter=SGD_iterations,
        random_state=41
    ),
    X=X_full,
    y=y_enc,
    train_sizes=sample_space,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=41),
    scoring='precision',
    n_jobs=-1
)

def make_tidy(sample_space, train_scores, valid_scores):
    """Transform a pivot-table-style dataset containing training and validation
    scores computed for plotting learning curves into tidy format where each 
    row is a training subset size and the columns are the specific training 
    subset sizes, training set scores, and validation set scores. This enables
    the data to be plotted with the Seaborn library.
    
    Args:
        sample_space (ndarray): a NumPy array of integers containing the
            sizes of the training subsets
        train_scores (ndarray): a NumPy array containing training set scores
            from 10 folds of cross-validation
        valid_scores (ndarray): a NumPy array containing validation set scores
            from 10 folds of cross-validation
    
    Returns:
        a Pandas DataFrame containing each training subset size and the mean
        training and validation score for each set size"""
    
    # Join train_scores and valid_scores, and label with sample_space
    messy_format = pd.DataFrame(
        np.stack((sample_space, train_scores.mean(axis=1),
                  valid_scores.mean(axis=1)), axis=1),
        columns=['# of training examples', 'Training set', 'Validation set']
    )
    
    # Re-structure table into into tidy format
    return pd.melt(
        messy_format,
        id_vars='# of training examples',
        value_vars=['Training set', 'Validation set'],
        var_name='Scores',
        value_name='precision'
    )

# Change the font scale
sns.set(font_scale=1.8)

# Initialize a FacetGrid object using the table of scores and facet on
# the score from the different sets
fig = sns.FacetGrid(
    make_tidy(sample_space, train_scores, valid_scores),
    hue='Scores',
    size=5,
    palette='Dark2',
    aspect=1.2
)

# Plot the learning curves, add a legend, and rescale y-axis
fig.map(plt.scatter, '# of training examples', 'precision', s=100)
fig.map(plt.plot, '# of training examples', 'precision')     .set(ylim=(0.5, 1.05), xticks=(0, 10000, 20000))     .add_legend();

# Save the figure
#fig.savefig('learning_curves.png', dpi=300, bbox_inches='tight');

# Initialize the hyperparameter space
param_dist = {
    'alpha': np.logspace(-6, -1, 50),
    'l1_ratio': np.linspace(0, 1, 50)
}

# Inner cross-validation loop to tune the hyperparameters
random_search = RandomizedSearchCV(
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

# Outer cross-validation loop to assess model performance
scores = cross_val_score(
    estimator=random_search,
    X=X_full,
    y=y_enc,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=41),
    scoring='precision'
)

# Display cross-validation scores
scores

# Display the average precision score and its standard deviation
print('Precision: {} +/- {}'.format(scores.mean(), scores.std()))

# Train the random hyperparameter search on the training set
random_search.fit(X_full, y_enc)

# Display the optimal hyperparameters
random_search.best_params_

# Train the classifier on the full dataset using the optimal hyperparameters
final_clf = SGDClassifier(
        loss='log',
        penalty='elasticnet',
        alpha=random_search.best_params_['alpha'],
        l1_ratio=random_search.best_params_['l1_ratio'],
        max_iter=SGD_iterations,
        random_state=41
)
final_clf.fit(X_full, y_enc);

# Select 10 different sizes of the dataset
sample_space = np.linspace(15, len(X_std) * 0.8, 10, dtype='int')

# Compute learning curves
train_sizes, train_scores, valid_scores = learning_curve(
    estimator=final_clf,
    X=X_full,
    y=y_enc,
    train_sizes=sample_space,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=41),
    scoring='precision',
    n_jobs=-1
)

# Initialize a FacetGrid object using the table of scores and facet on
# the score from the different sets
fig = sns.FacetGrid(
    make_tidy(sample_space, train_scores, valid_scores),
    hue='Scores',
    size=5,
    palette='Dark2',
    aspect=1.2
)

# Plot the learning curves, add a legend, and rescale y-axis
fig.map(plt.scatter, '# of training examples', 'precision', s=100)
fig.map(plt.plot, '# of training examples', 'precision')     .set(ylim=(0.5, 1.05), xticks=(0, 10000, 20000))     .add_legend();

# Combine meta feature labels with n-gram labels
all_features = features + vectorizer.get_feature_names()

# Add the corresponding feature names to the parameters, sorted from highest
# to lowest
feature_ranks = pd.Series(
    final_clf.coef_.T.ravel(),
    index=all_features
).sort_values(ascending=False)[:19][::-1]

# Display a bar graph of the top features
graph = feature_ranks.plot(
    kind='barh',
    legend=False,
    figsize=(4, 8),
    color='#666666'
);

# Save the figure
#fig = graph.get_figure()
#fig.savefig('ngrams', dpi=300, bbox_inches='tight');

# Add the corresponding meta feature names to the parameters, sorted from
# highest to lowest
meta_feature_ranks = pd.Series(
    final_clf.coef_.T.ravel()[:len(features)],
    index=features
).sort_values(ascending=False)[::-1]

# Display a bar plot of the meta feature importance
graph2 = meta_feature_ranks.plot(
    kind='barh',
    legend=False,
    figsize=(5, 8),
    color='#666666'
)

# Change the meta feature labels from variables to names
labels = [
    '# of hyperlinks',
    '# of images',
    '# of innovation words',
    '# of exclamation marks',
    '% of text bolded', 
    '# of words',
    '# of YouTube videos',
    '% of exclamation marks',
    '# of sentences',
    '# of GIFs',
    '% innovation words',
    '# of all-caps words',
    '# of bold tags',
    '% all-caps words',
    '# of videos',
    'Avg words/paragraph',
    '# of paragraphs',
    'Avg words/sentence',
    'Avg sentences/paragraph'    
]
plt.yticks(np.arange(19), labels[::-1]);

# Save the figure
#fig2 = graph2.get_figure()
#fig2.savefig('meta', dpi=300, bbox_inches='tight');

# Select a hyperlink
hyperlink = 'https://www.kickstarter.com/projects/getpebble/pebble' +     '-2-time-2-and-core-an-entirely-new-3g-ultra'

# Compute the meta features and preprocess the campaign section
meta_features, processed_section = feature_engineering.process_project(
    hyperlink
)

# Compute the n-grams from the preprocessed text
ngrams = vectorizer.transform([processed_section])

# Standardize the meta features and convert results into a sparse matrix
scaled_meta_features = sparse.csr_matrix(scaler.transform([meta_features]))

# Concatenate the meta features with the n-gram vector
feature_vector = sparse.hstack([scaled_meta_features, ngrams])

# Display the probability of being funded
final_clf.predict_proba(feature_vector)[0, 1]

