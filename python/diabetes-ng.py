# Major libraries
import scipy
import bottleneck # for speeding up pandas operations
import numexpr # ditto
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Mining / EDA / dimensionality reduction
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import scale
from scipy.spatial.distance import euclidean

# Supervised learning
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import train_test_split, cross_val_score, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Unsupervised learning
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib.colors import ListedColormap

get_ipython().magic('matplotlib inline')
rng = np.random.RandomState(1)

print(pd.__version__)

Dataset = {} # I found it was much easier to manage multiple datasets by keeping them all in a dictionary
Dataset['Raw'] = pd.read_csv('./data/diabetic_data.csv', index_col='encounter_id', na_values="?", low_memory=False)
Dataset['Raw'].shape

# Print contents of dataset
Dataset['Raw'].head(5)

# Get an idea of how many features are missing values, and how many values they're missing:
def percent_null(data):
    # Returns a Pandas series of what percentage of each feature of 'data' contains NaN values
    pc_null = data.apply(pd.Series.isnull).apply(lambda x: 100*round(len(x[x==True])/len(x), 4))
    return pc_null[pc_null!=0]
percent_null(Dataset['Raw'])

feature_value_counts = [] # A list to put Series containing the number of entries for each level of a feature
for feature in Dataset['Raw'].columns:
    feature_value_counts.append(Dataset['Raw'][feature].value_counts())

feature_value_counts[32].ix[:] # Manually iterated through each feature to check for typos / misentries

Dataset['Datatyped'] = Dataset['Raw'].copy() # To allow comparison between datasets before and after modification

# Remove useless features
Dataset['Datatyped'].drop(['weight', 'payer_code'], axis=1, inplace=True);

# Label-encode age feature to an integer in the center of the raw bin
Dataset['Datatyped'].age = (LabelEncoder().fit_transform(Dataset['Datatyped'].age)*10) + 5 

# Convert features to appropriate datatype - nominal and ordinate variables as categorical dtypes, interval variables as integers
cat_features = ['patient_nbr', 'race', 'gender', 'admission_type_id',
       'discharge_disposition_id', 'admission_source_id',
       'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
       'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'change', 'diabetesMed', 'max_glu_serum', 'A1Cresult', 'readmitted']
num_features = Dataset['Datatyped'].columns.drop(cat_features).values

for feature in cat_features:
    Dataset['Datatyped'][feature] = Dataset['Datatyped'][feature].astype('category')

for feature in num_features:
    Dataset['Datatyped'][feature] = Dataset['Datatyped'][feature].astype('int')

Dataset['Datatyped'].dtypes # Check casting was successful

#It should be possible to automatically datatype these using built-in pandas functions

print(pd.__version__)

pd.set_option('max_columns',40) # To allow all columns to be seen
Dataset['Datatyped'][cat_features].describe(exclude = None) # Categorical features description

Dataset['Datatyped'][num_features].describe(exclude = None).round(1) # Numerical features description

pd.set_option('max_columns',10)

# Capture the first entry for each patient in a new 'Independent' entries dataset
def one_entry(data):
    unique_entry = data.iloc[0,:]
    return unique_entry

Dataset['Independent'] = Dataset['Datatyped'].copy()
Dataset['Independent'] = Dataset['Independent'].groupby('patient_nbr').apply(one_entry)

Dataset['Independent'].index = Dataset['Independent']['patient_nbr']
Dataset['Independent'].drop('patient_nbr', axis=1, inplace=True)

Dataset['Independent'].head(3)

Dataset['Independent'].index.value_counts()[0:5] # Check that top value counts of patient numbers are equal to one

Dataset['Encoded response'] = Dataset['Independent'].copy()
le_readm = LabelEncoder()
Dataset['Encoded response'].readmitted = pd.Series(le_readm.fit_transform(Dataset['Encoded response'].readmitted),
                                                      index=Dataset['Encoded response'].index)
le_readm.classes_ # 0 = <30, 1 = >30, 2 = NO

Dataset['Encoded response'].readmitted.value_counts()

# Check which features contain null values
percent_null(Dataset['Encoded response'])

# Impute for missing values by response class
for response in Dataset['Encoded response']['readmitted'].unique():
    response_df = Dataset['Encoded response'][Dataset['Encoded response'].readmitted == response].copy()
    
    # Impute for medical specialties
    response_df = response_df.sort_values(['diag_1', 'age', 'admission_source_id']) # Clusters similar medical specialties
    response_df['medical_specialty'] = response_df['medical_specialty'].fillna(method='bfill') # backward-fill null values
    
    # Impute for race
    response_df['race'] = response_df['race'].fillna(value=response_df['race'].describe().top) # Mode vaue for race
    
    # Impute for diagnoses
    response_df['diag_1'] = response_df['diag_1'].fillna(value=response_df['diag_1'].describe().top)
    response_df['diag_2'] = response_df['diag_2'].fillna(value=response_df['diag_2'].describe().top) # Mode values
    response_df['diag_3'] = response_df['diag_3'].fillna(value=response_df['diag_3'].describe().top)
    
    # Insert imputed data back into main DataFrame
    Dataset['Encoded response'][Dataset['Encoded response'].readmitted == response] = response_df

# Check that imputation was successful - we're expecting percent_null() to return an empty Series
percent_null(Dataset['Encoded response'])

def ICD9_classifier(diagnoses):
    # Returns a series of strings corresponding to type of ICD9 diagnosis
    # diagnoses is a list
    gr_diagnoses = diagnoses.copy()
    icd9_dct = {
                'Infectious':(1, 139),
                'Neoplasmic':(140,239),
                'Hormonal':(240, 279),
                'Blood':(280,289),
                'Mental':(290,319),
                'Nervous':(320,359),
                'Sensory':(360,389),
                'Circulatory':(390,459),
                'Respiratory':(460,519),
                'Digestive':(520,579),
                'Genitourinary':(580,629),
                'Childbirth':(630,679),
                'Dermatological':(680,709),
                'Musculoskeletal':(710,739),
                'Congenital':(740,759),
                'Perinatal':(760,779),
                'Miscellaneous':(780,799),
                'Injury':(800,999)
               }
    for i, diagnosis in enumerate(diagnoses):
        if (str(diagnoses[i])[0] == 'E') or (str(diagnoses[i])[0] == 'V'):
            gr_diagnoses[i] = 'Accidental'
        elif (str(diagnoses[i]).lower() == 'nan'):
            gr_diagnoses[i] = 'NaN'
        else:
            for key, icd_range in icd9_dct.items():
                if (int(float(diagnoses[i])) >= icd_range[0]) and (int(float(diagnoses[i])) <= icd_range[1]):
                    gr_diagnoses[i] = key
    return gr_diagnoses

d1 = ICD9_classifier(Dataset['Encoded response'].diag_1.values)
d2 = ICD9_classifier(Dataset['Encoded response'].diag_2.values)
d3 = ICD9_classifier(Dataset['Encoded response'].diag_3.values)

Dataset['Aggregated diagnoses'] = Dataset['Encoded response'].copy()

Dataset['Aggregated diagnoses'].diag_1 = d1
Dataset['Aggregated diagnoses'].diag_2 = d2
Dataset['Aggregated diagnoses'].diag_3 = d3

Dataset['Aggregated diagnoses'].head(5)

Dataset['Aggregated diagnoses'].shape # Before removing outliers

Dataset['Trimmed'] = Dataset['Aggregated diagnoses'].copy()
Dataset['Trimmed'].describe().ix[['min', 'max'],:]

# Remove outliers by class
for response in Dataset['Trimmed'].readmitted.unique(): # For each readmission response
    response_df = Dataset['Trimmed'][Dataset['Trimmed'].readmitted==response] # For all samples that meet the readmission criteria
    for column in response_df[num_features].columns: # Iterate through each feature of the sample
        Q1 = response_df[column].quantile(0.25)
        Q3 = response_df[column].quantile(0.75)
        IQR = Q3-Q1
        if IQR > 0: # And remove features that are outside of Q1 - 1.5IQR or Q3 + 1.5IQR
            response_df = response_df[(response_df[column] > Q1-(1.5*IQR)) & (response_df[column] < Q3+(1.5*IQR))]
            Dataset['Trimmed'] = Dataset['Trimmed'].drop(Dataset['Trimmed'][Dataset['Trimmed'].readmitted==response].index, axis=0)
            Dataset['Trimmed'] = Dataset['Trimmed'].append(response_df)

Dataset['Trimmed'].describe().ix[['min', 'max'],:]

Dataset['Trimmed'].shape # After outlier-removal

# Separate response from features
y = Dataset['Trimmed'].readmitted
X = Dataset['Trimmed'][(Dataset['Trimmed'].columns).drop('readmitted')]

cat_features.remove('readmitted'); cat_features.remove('patient_nbr')

X_ohe = pd.get_dummies(X, columns=cat_features)
Dataset['Large OHE'] = X_ohe.join(y) # One-hot encoded dataset with many dimensions
Dataset['Large OHE'].shape

correlations_df = Dataset['Large OHE'].corr()

plt.figure(figsize=(15,40))
abs_corr_coef = abs(correlations_df.ix[correlations_df.columns.drop('readmitted'),'readmitted'].sort_values())
sns.barplot(abs_corr_coef, abs_corr_coef.index, orient='h', palette=sns.color_palette('coolwarm', len(abs_corr_coef)))
plt.ylabel('Predictor label'); plt.xlabel('Absolute correlation with readmission');
plt.title('Distribution of absolute feature correlation coefficients with readmission status', size=16);

num_features_corr = Dataset['Large OHE'][num_features].corr()
plt.figure(figsize=(15,8))
sns.heatmap(num_features_corr, annot=True, cmap='seismic')
plt.xticks(rotation=90); plt.title('Heatmap of numerical feature Pearson correlation coefficients', size=14);

g = sns.FacetGrid(Dataset['Trimmed'], col="readmitted", subplot_kws={'alpha':1}, size=6,
                  sharey=False, palette='colorblind',hue='readmitted')  
g.map(sns.distplot, "number_inpatient", kde=False, hist_kws={'width':0.5, 'alpha':1, 'align':'mid'},
      bins=np.arange(-0.25,7.25,1));
g.axes[0][0].set_ylabel('Number of patients')
plt.suptitle('Frequency distribution of inpatient encounters by readmission status', size=16);
plt.subplots_adjust(top=0.9);

g = sns.FacetGrid(Dataset['Trimmed'], col="readmitted", subplot_kws={'alpha':1}, size=5, sharey=False, palette='colorblind',
                  hue='readmitted')  
g.map(sns.distplot, "time_in_hospital", kde=False, hist_kws={'width':0.5, 'alpha':1}, bins=np.arange(-0.5,17));
g.axes[0][0].set_ylabel('Number of patients')
plt.suptitle('Frequency distribution of patient time in hospital by readmission status', size=16);
plt.subplots_adjust(top=0.85);

f = plt.figure(figsize=(13,5))
sns.boxplot(x='readmitted', y='time_in_hospital',
           data=Dataset['Trimmed'], palette='colorblind', saturation=1, orient='v', notch=True)
plt.ylabel('Time in hospital [days]', size=12); plt.xlabel('Readmitted within 30 days [T/F]', size=12);
plt.title('Time spent in hospital by readmission status', size=16)
plt.tight_layout()

g = sns.FacetGrid(Dataset['Aggregated diagnoses'], col="readmitted", subplot_kws={'alpha':1}, size=6, sharey=False,
                  palette='colorblind',hue='readmitted')  
g.map(sns.distplot, "age", kde=False, hist_kws={'width':9, 'alpha':1}, bins=np.arange(0,100,10));
g.axes[0][0].set_ylabel('Number of patients')
plt.suptitle('Frequency distribution of age by readmission status', size=14);
plt.subplots_adjust(top=0.9);

plt.figure(figsize=(12, 5))
sns.boxplot(x='readmitted', y='age', data=Dataset['Encoded response'], palette='colorblind', notch=True, saturation=1)
plt.tight_layout()
plt.title('Patient age frequencies by readmission status', size=14);

Dataset['Large OHE']['age'][Dataset['Trimmed']['readmitted']==0].mean()

Dataset['Large OHE']['age'][Dataset['Trimmed']['readmitted']==1].mean()

Dataset['Large OHE']['age'][Dataset['Trimmed']['readmitted']==2].mean()

plt.figure(figsize=(15,5))
Dataset['Trimmed'].diag_1.value_counts().plot(color=sns.color_palette('colorblind')[1], kind='bar', rot=90);
plt.title('Primary diagnoses occurences by dignosis type', size=16);
plt.ylabel('Occurences [#]');

# Response class centroid pairwise distances
Dataset['Scaled'] = pd.DataFrame(scale(Dataset['Large OHE']), index=Dataset['Large OHE'].index, columns=Dataset['Large OHE'].columns)

centroid_0 = Dataset['Scaled'].loc[Dataset['Large OHE'].readmitted == 0, :].mean()
centroid_1 = Dataset['Scaled'].loc[Dataset['Large OHE'].readmitted == 1, :].mean()
centroid_2 = Dataset['Scaled'].loc[Dataset['Large OHE'].readmitted == 2, :].mean();

euclidean(centroid_1, centroid_2) # '>30 days' and 'No' were slightly closer

euclidean(centroid_0, centroid_1) # but EDA suggested '<30 days' and '>30 days' were closer for the most important features

# 3 different datasets to test classifier performances on
Dataset['Large supervised'] = Dataset['Large OHE'].copy() # Includes all features
Dataset['Filtered supervised'] = Dataset['Large OHE'].copy() # Will undergo filtering
Dataset['Dense supervised'] = Dataset['Trimmed'][np.concatenate((num_features, ['readmitted']), 0)] # Only includes num. features

# Create a binary response - readmitted = {True, False}, removed > 30 days
for prefix in ['Large', 'Filtered', 'Dense']:
   set_label = prefix + ' supervised'
   Dataset[set_label].loc[Dataset[set_label].readmitted == 0, 'readmitted'] = 1 # {0, 1} -> {1}
   Dataset[set_label].loc[Dataset[set_label].readmitted == 2, 'readmitted'] = 0 # {2} -> {0}

# Split filtered data into predictors and response
X = Dataset['Filtered supervised'].loc[:, Dataset['Filtered supervised'].columns.drop('readmitted')]
y = Dataset['Filtered supervised'].loc[:, 'readmitted']

# One-hot encode predictors
ohe_features = pd.get_dummies(X[X.columns.drop(num_features)])
X = X[num_features].join(ohe_features)

# Split all datasets into training and testing subsets
Training = {}; Testing = {};

(Training['O_Filtered'], Testing['O_Filtered'],
Training['R_Filtered'], Testing['R_Filtered']) = train_test_split(X, y, random_state=0)

(Training['O_Large'], Testing['O_Large'],
 Training['R_Large'], Testing['R_Large'] ) = train_test_split(Dataset['Large supervised'].drop('readmitted',axis=1),
                                                                Dataset['Large supervised'].readmitted, random_state=0)

(Training['O_Dense'], Testing['O_Dense'],
 Training['R_Dense'], Testing['R_Dense'] ) = train_test_split(Dataset['Dense supervised'].drop('readmitted',axis=1),
                                                                Dataset['Dense supervised'].readmitted, random_state=0)

opt_metric = metrics.make_scorer(metrics.matthews_corrcoef) # Will be explained later

pca = PCA()
ss = StandardScaler()

for set_label in ['Large', 'Filtered', 'Dense']:
    obs_label = 'O_'+set_label
    Training[obs_label] = pd.DataFrame(ss.fit_transform(Training[obs_label]), index=Training[obs_label].index,
                                            columns=Training[obs_label].columns)
    Testing[obs_label] = pd.DataFrame(ss.fit_transform(Testing[obs_label]), index=Testing[obs_label].index,
                                         columns=Testing[obs_label].columns)

pc_labels = ['PC_'+str(i) for i in range(0,len(Training['O_Filtered'].columns))]
Training['O_Filtered'] = pd.DataFrame(pca.fit_transform(Training['O_Filtered']),
                                      index=Training['O_Filtered'].index, columns=pc_labels)
Testing['O_Filtered'] = pd.DataFrame(pca.transform(Testing['O_Filtered']), index=Testing['O_Filtered'].index,
                                     columns=pc_labels)

# Examine initial distribution of principal component variances
vt = VarianceThreshold()
vt.fit(Training['O_Filtered'])
plt.figure(figsize=(15,5))
plt.plot(np.arange(0,len(Training['O_Filtered'].columns)),pd.Series(vt.variances_), color=sns.color_palette('colorblind')[1])
plt.xlabel('Principle component #'); plt.ylabel('Variance in component dimension');
plt.title('Principle component variances before thresholding', size=14);

# Create a function to examine how a classifier's behaviour changes with the number of components
def optimise_variance_threshold(classifier, X, y, max_variance, increment, scoring_metric, min_variance=0):
    scores = []
    for current_threshold in np.arange(min_variance, max_variance, increment):
        vt_0 = VarianceThreshold(threshold=current_threshold)
        X_vt = pd.DataFrame(vt_0.fit_transform(X), index=X.index)
        variance_score = cross_val_score(classifier, X_vt, y, scoring=scoring_metric)
        scores.append(variance_score.mean())
    return scores

variance_mcc_scores = optimise_variance_threshold(LinearDiscriminantAnalysis(), Training['O_Filtered'],
                                                  Training['R_Filtered'], scoring_metric=opt_metric, max_variance=3,
                                                  increment=0.05, min_variance=0.01)

plt.figure(figsize=(15,6));
plt.plot(np.arange(0, 3, 0.05), variance_mcc_scores, color=sns.color_palette('colorblind')[2])
plt.xlabel('Variance threshold'); plt.ylabel('Score'); plt.title('LDA Matthews correlation coefficient scores by variance threshold', size=16);
plt.legend()
plt.show()

# Remove noisy components
vt_0 = VarianceThreshold(threshold=0.6)
Training['O_Filtered'] = pd.DataFrame(vt_0.fit_transform(Training['O_Filtered']), index=Training['O_Filtered'].index)
Testing['O_Filtered'] = pd.DataFrame(vt_0.transform(Testing['O_Filtered']), index=Testing['O_Filtered'].index)

column_labels = ['PC_'+str(i) for i in range(0, len(Training['O_Filtered'].columns))]
Training['O_Filtered'].columns = column_labels
Testing['O_Filtered'].columns = column_labels

Training['O_Filtered'].shape

# Null classifier against which to compare all others
dummy_classifier = DummyClassifier(strategy='stratified')
dummy_classifier.fit(Training['O_Filtered'], Training['R_Filtered'])
dummy_predictions = cross_val_predict(dummy_classifier, Training['O_Filtered'], 
                                      Training['R_Filtered'], cv=5)
null_classifier_score = cross_val_score(dummy_classifier, Training['O_Filtered'],
                                        Training['R_Filtered'], scoring=opt_metric, cv=5).mean()
null_classifier_score

# Unoptimised classifiers - effectively a feature set comparison
LDA = {}
LDA_scores = {}
for set_label in ['Large', 'Filtered', 'Dense']:
    LDA[set_label] = LinearDiscriminantAnalysis()
    LDA_scores[set_label] = cross_val_score(LDA[set_label], X=Training['O_'+set_label],
                                            y=Training['R_'+set_label], cv=5, scoring=opt_metric)

LDA_scores

prior_values = []
for i in np.arange(0, 1, 0.01):
    prior_values.append([i, 1-i])

lda_gridCV = GridSearchCV(LinearDiscriminantAnalysis(),
                          param_grid = {'priors':prior_values,
                                        'solver':['svd', 'lsqr'], # NumPy wasn't able to solve the eigenvector problem for this dataset?
                                        },
                          scoring=opt_metric)

lda_gridCV.fit(Training['O_Filtered'], Training['R_Filtered']);

svd_scores = []; lsqr_scores = [];

for classifier_info in lda_gridCV.grid_scores_:
    if classifier_info[0]['solver'] == 'svd':
        svd_scores.append(classifier_info[1])
    if classifier_info[0]['solver'] == 'lsqr':
        lsqr_scores.append(classifier_info[1])

plt.figure(figsize=(15, 7))
solver_scores = {'svd':svd_scores, 'lsqr':lsqr_scores}
i=0
for key, scores_vector in solver_scores.items():
    plt.plot(np.arange(0,1,0.01), scores_vector, color=sns.color_palette('colorblind')[i], label=key)
    i+=1

plt.legend(loc=2); plt.title('LDA score by majority class prior and solver type', size=14)
plt.xlabel('Specified majority class prior probability')
plt.ylabel('Matthews corr. coef.')
plt.show()

optimum_priors = lda_gridCV.best_params_['priors']
optimum_priors

# Shrinkage optimisation
lda_shrinkage_gridCV = GridSearchCV(LinearDiscriminantAnalysis(solver='lsqr', priors=optimum_priors),
                                    param_grid = {'shrinkage':np.arange(0,1,0.01)},
                                    scoring=opt_metric)
lda_shrinkage_gridCV.fit(Training['O_Filtered'], Training['R_Filtered']);

shrinkage_scores = []
for shrinkage_result in lda_shrinkage_gridCV.grid_scores_:
    shrinkage_scores.append(shrinkage_result[1])

plt.figure(figsize=(15,7))
plt.plot(np.arange(0,1,0.01), shrinkage_scores, color=sns.color_palette('deep')[2])
plt.xlabel('Shrinkage value'); plt.ylabel('Matthews corr. coef.')
plt.title('LDA MCC scores against L1-shrinkage values', size=14);

optimum_shrinkage = lda_shrinkage_gridCV.best_params_['shrinkage']
optimised_lda = LinearDiscriminantAnalysis(solver='lsqr', priors=optimum_priors, shrinkage=optimum_shrinkage)
optimised_lda.fit(Training['O_Filtered'], Training['R_Filtered']);

# List of regularisation strengths
C_values = np.concatenate([np.arange(0.001,0.01,0.001), np.arange(0.01,0.1,0.01), np.arange(0.1,1,0.1)])

# Optimise a LR classifier by varying the reg. strength
lr_gridCV = GridSearchCV(LogisticRegression(penalty='l1', solver='liblinear', warm_start=True, random_state=0),
                        param_grid={'C':C_values}, scoring=opt_metric)
lr_gridCV.fit(Training['O_Large'], Training['R_Large']);

lr_scores = []
for scores in lr_gridCV.grid_scores_: lr_scores.append(scores[1])

# Display classifier performance by reg. strength
plt.figure(figsize=(14,7))
plt.plot(C_values, lr_scores, marker='o', color=sns.color_palette('colorblind')[1])
plt.title('Correlation between LR predictions and actual responses by regularisation strength', size=16)
plt.xscale('log'); plt.xlabel('$1/\lambda$ for lasso shrinkage');
plt.ylabel('Matthews correlation coefficient');

# Extract coefficients of best LR model
lr_coefficients = pd.Series(lr_gridCV.best_estimator_.coef_.flatten(), index=Training['O_Large'].columns)

# Visualise coefficient sizes in a bar chart to show relative importance to decision boundary
plt.figure(figsize=(15,5))
lr_coefficients.loc[abs(lr_coefficients)>0].sort_values(ascending=False).head(40).plot(kind='bar', color=sns.color_palette('colorblind')[1]);
plt.xticks(rotation=90);

lr = LogisticRegression(penalty='l1', C=0.001)
lr.fit(Training['O_Large'], Training['R_Large'])
shrunk_coefficients = pd.Series(lr.coef_.flatten(), index=Training['O_Large'].columns)
shrunk_coefficients = shrunk_coefficients.loc[abs(shrunk_coefficients)>0].sort_values(ascending=False)

shrunk_coefficients

correlations_df['readmitted'].sort_values()[0:8]

correlations_df['readmitted'].sort_values(ascending=False)[0:4]

# Smaller datasets to train KNN with, based on remaining coefficients after shrinkage on a LR classifier
Training['O_KNN'] = Training['O_Large'].ix[:, shrunk_coefficients.index]
Training['R_KNN'] = Training['R_Large'].copy()
Testing['O_KNN'] = Testing['O_Large'].ix[:, shrunk_coefficients.index]
Testing['R_KNN'] = Testing['R_Large'].copy()

knn_gridCV = GridSearchCV(KNeighborsClassifier(), param_grid={'n_neighbors':np.arange(5, 51, 5),
                                                              'weights':['uniform']}, scoring=opt_metric)
knn_gridCV.fit(Training['O_KNN'], Training['R_KNN']);

def plot_gridSearch_performance(grid_scores):
    param_values = []; scores_mu = []
    for i in range(0, len(grid_scores)):
        param_values.append(grid_scores[i][0]['n_neighbors'])
        scores_mu.append(grid_scores[i][1])
    plt.figure(figsize=(15, 5))
    plt.plot(param_values, scores_mu, linestyle='None', marker='o')

plot_gridSearch_performance(knn_gridCV.grid_scores_)

knn_predictions = cross_val_predict(knn_gridCV, Training['O_KNN'], Training['R_KNN'])

print(metrics.confusion_matrix(Training['R_KNN'], knn_predictions))

Predictions = {}
Predictions['LDA'] = optimised_lda.predict(Testing['O_Filtered'])
Predictions['KNN'] = knn_gridCV.predict(Testing['O_KNN'])
Predictions['Dummy'] = dummy_classifier.predict(Testing['O_Filtered'])

print(metrics.classification_report(Testing['R_Filtered'], Predictions['LDA']))

print(metrics.classification_report(Testing['R_KNN'], Predictions['KNN']))

print(metrics.classification_report(Testing['R_Filtered'], Predictions['Dummy']))

print(metrics.confusion_matrix(Testing['R_Filtered'], Predictions['LDA']))

print(metrics.confusion_matrix(Testing['R_KNN'], Predictions['KNN']))

print(metrics.accuracy_score(Testing['R_Filtered'], Predictions['LDA']))

print(metrics.accuracy_score(Testing['R_KNN'], Predictions['KNN']))

print(metrics.accuracy_score(Testing['R_KNN'], Predictions['Dummy']))

sns.lmplot(data=Testing['O_Filtered'].join(pd.Series(Predictions['LDA'], index=Testing['O_Filtered'].index, name='Predictions')),
           x='PC_0', y='PC_1', size=6, aspect=1.618, fit_reg=False, hue='Predictions', palette='colorblind', col='Predictions');

# PC 1 loadings
PC_1_loadings = pd.DataFrame(pca.components_, index=pc_labels,
             columns=Dataset['Large OHE'].columns.drop('readmitted')).loc['PC_1',:]
PC_1_loadings[abs(PC_1_loadings)>0.1].sort_values()

def forward_stepwise_selection(X_train, y_train, termination_step=len(X_train.columns.values), score_metric='f1', class_weights='balanced'):
    # Returns a list of feature labels that are in order of how much they improve a logistic regression classifier
    # Uses k-fold cross validation with k = 5
    p = len(X_train.columns) # Number of features
    remaining_features = (X_train.columns.values).tolist()
    selected_features = []
    trialled_features = []
    step_scores = []
    for k in range(0, termination_step): # Iterate as many times as there are features
        step_max_score = 0
        for j in range(0, p-k): # For each of the features that haven't yet been added to the model
            trialled_features = selected_features+[remaining_features[j]]
            cross_validated_score = cross_val_score(LogisticRegression(class_weight=class_weights),
                                                 X_train[trialled_features], y_train, cv=5, scoring=score_metric)
            trialled_classifier_score = cross_validated_score.mean()
            if trialled_classifier_score > step_max_score: # See which improves the model's accuracy the most
                step_max_score = trialled_classifier_score
                step_feature_index = j
        best_feature_of_step = remaining_features[step_feature_index]
        selected_features.append(best_feature_of_step)  # Then add that feature to the model
        del remaining_features[step_feature_index] # And remove it from the list of features that still need to be added
        step_scores.append(step_max_score) # Make a record of the model's accuracy with this number of features
        print("Completed step ", str(k), ", score is ", round(step_max_score, 2))
    return (selected_features, step_scores)

Dataset['Unsupervised'] = pd.concat([Training['O_Filtered'].loc[:,:],
                                     Testing['O_Filtered'].loc[:,:]])

pc_50_labels = ['PC_'+str(j) for j in range(0, 51)]

# Some algorithms used demand alot of memory, so we'll need to use a smaller dataset for those
chosen_indices = np.random.choice(Dataset['Unsupervised'].index, 1000, replace=False)
Dataset['Unsupervised small'] = Dataset['Unsupervised'].ix[chosen_indices, pc_50_labels]

plt.figure(figsize=(15,5))
sns.regplot(x='PC_0', y='PC_1', data=Dataset['Unsupervised'], color=sns.color_palette('colorblind')[1], fit_reg=False);
plt.title('Projection of dataset onto first two principal components');

KMClusterers = {} # what an inelegant word
KMClusterers['Parent'] = KMeans(n_clusters=2, random_state=0, max_iter=300)

Clusters = {}
Clusters['KM parent'] = pd.Series(KMClusterers['Parent'].fit_predict(Dataset['Unsupervised']),
                              index=Dataset['Unsupervised'].index, name='KM parent')

sns.lmplot(x='PC_0', y='PC_1', col='KM parent', data=Dataset['Unsupervised'].join(Clusters['KM parent']), palette='colorblind',
           hue='KM parent', fit_reg=False, size=5, aspect=1.4);
plt.title('Projection of dataset onto first two principal components');

Dataset['Unsupervised KM subset'] = Dataset['Unsupervised'].loc[Clusters['KM parent']==0, :]

KMClusterers['Child'] = KMeans(n_clusters=2, random_state=0, max_iter=300)
Clusters['KM child'] = pd.Series(KMClusterers['Child'].fit_predict(Dataset['Unsupervised KM subset']),
                                 index=Dataset['Unsupervised KM subset'].index, name='KM child');

Clusters['KM'] = Clusters['KM child'] + 2 
Clusters['KM'] = pd.concat([Clusters['KM'],
               Clusters['KM parent'].loc[Clusters['KM parent'].index.drop(Clusters['KM child'].index)]],
               axis=0)
Clusters['KM'].name = 'KM clusters'

sns.lmplot(x='PC_0', y='PC_1', data=Dataset['Unsupervised'].join(Clusters['KM']),
           palette='colorblind', hue='KM clusters', fit_reg=False, size=5, aspect=2.5);
plt.title('Projection of dataset onto the first two principal components');

# Make datasets consisting of samples from each cluster
Dataset['KM 1'] = Dataset['Trimmed'].loc[Clusters['KM']==1, :]
Dataset['KM 2'] = Dataset['Trimmed'].loc[Clusters['KM']==2, :]
Dataset['KM 3'] = Dataset['Trimmed'].loc[Clusters['KM']==3, :]

# Create a dataframe containing the means of each feature for each of the above datasets
k1_mus = Dataset['KM 1'].describe().ix['mean',:]
k2_mus = Dataset['KM 2'].describe().ix['mean',:]
k3_mus = Dataset['KM 3'].describe().ix['mean',:]
kn_mus = pd.concat([k1_mus, k2_mus, k3_mus], axis=1)

kn_mus.columns = ['Cluster 1', 'Cluster 2', 'Cluster 3']

kn_mus = kn_mus.apply(func=(lambda row: row/(row.mean())), axis=1)

cblind = [sns.color_palette('colorblind')[0],sns.color_palette('colorblind')[1],sns.color_palette('colorblind')[2]] 

kn_mus.plot.bar(cmap=ListedColormap(cblind), figsize=(15, 7), title='Cluster feature means');

kn_mus.apply(func=(lambda row: row/(row.mean())), axis=1) # Scaling the means to better represent cluster differences

PC_0_loadings = pd.DataFrame(pca.components_, index=pc_labels,
             columns=Dataset['Large OHE'].columns.drop('readmitted')).loc['PC_0',:]
PC_0_loadings[abs(PC_0_loadings)>0.1].sort_values()

def normalised_value_counts(series):
    return series.value_counts()/len(series)

f, ax = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(16,6))
N_bars = np.arange(len(Dataset['KM 1'].diag_1.unique()))*2
for i in range(0,3):
    normalised_value_counts(Dataset['KM '+str(i+1)].diag_1).plot(kind='barh', ax=ax[i],
                                                                 color=sns.color_palette('colorblind')[i])
    ax[i].set_title('Cluster '+str(i+1))
    
plt.tight_layout()

Diagnoses = {}
for i in range(0,3):
    Diagnoses['Cluster '+str(i+1)] = pd.to_numeric(Dataset['Encoded response'].diag_1.loc[Clusters['KM'].loc[Clusters['KM']==i+1].index],
                                                   errors='coerce')
    Diagnoses['Cluster '+str(i+1)].dropna()
    
c1_hormonal_diag = (Diagnoses['Cluster 1'] < 280) & (Diagnoses['Cluster 1'] >= 240)
c1_digestive_diag = (Diagnoses['Cluster 1'] < 580) & (Diagnoses['Cluster 1'] >= 520)
c2_hormonal_diag = (Diagnoses['Cluster 2'] < 280) & (Diagnoses['Cluster 2'] >= 240)
c2_digestive_diag = (Diagnoses['Cluster 2'] < 580) & (Diagnoses['Cluster 2'] >= 520)

# ICD9 hormonal diagnosis codes by count for cluster 1
Diagnoses['Cluster 1'].loc[c1_hormonal_diag].value_counts().head(5)

# ICD9 hormonal diagnosis codes by count fir cluster 2
Diagnoses['Cluster 2'].loc[c2_hormonal_diag].value_counts().head(5)

# ICD9 digestive diagnosis codes by count for cluster 1
Diagnoses['Cluster 1'].loc[c1_digestive_diag].value_counts().head(5)

# ICD9 digestive diagnosis codes by count for cluster 2
Diagnoses['Cluster 2'].loc[(Diagnoses['Cluster 2'] < 580) & (Diagnoses['Cluster 2'] >= 520)].value_counts().head(5)

normalised_value_counts(Dataset['KM 1'].readmitted)

normalised_value_counts(Dataset['KM 2'].readmitted)

for p in np.arange(1, 101, 5):
    tsne = TSNE(n_components=2, perplexity=p, random_state=0, early_exaggeration=10, init='pca')
    Dataset['t-SNE p='+str(p)] = pd.DataFrame(tsne.fit_transform(Dataset['Unsupervised small']),
                                                 index = Dataset['Unsupervised small'].index,
                                                 columns=['Axis 0', 'Axis 1'])

# Your initialisation
f, ax = plt.subplots(4, 5, sharex=False, sharey=False, figsize=(16,14))
for i, k in enumerate(np.arange(1,101, 5)):
    j = int(i/5)
    i = i - (int(i/5)*5)
    Dataset['t-SNE p='+str(k)].plot(kind='scatter', x='Axis 0', y='Axis 1',
                                    ax=ax[j, i], color=sns.cubehelix_palette(20, start=.5, rot=-.75)[int(k/5)])
    ax[j,i].set_title('p = '+str(k))
    
plt.tight_layout()

Dataset['t-SNE'] = Dataset['t-SNE p=56']

# Your distances distribution
distances = pairwise_distances(Dataset['t-SNE'])
neighbor_distances = [np.min(row[np.nonzero(row)]) for row in distances]

plt.figure(figsize=(14, 6))
sns.distplot(neighbor_distances, color=sns.color_palette('colorblind')[1]);
plt.title('Mapped dataset nearest-neighbor distance frequency distribution');
plt.xlabel('Distance'); plt.ylabel('Frequency')

nb_clusters = []
for eps_value in np.arange(0.1, 10, 0.01):
    db = DBSCAN(eps=eps_value, min_samples=20)
    Clusters['DB'] = pd.Series(db.fit_predict(Dataset['t-SNE']), index=Dataset['t-SNE'].index, name='DBSCAN clusters')
    nb_clusters.append(len(Clusters['DB'].unique()))

# Your cluster counts with eps
plt.figure(figsize=(15,5))
plt.plot(np.arange(0.1, 10, 0.01), nb_clusters, color=sns.color_palette('colorblind')[1]);
plt.xlabel('eps value'); plt.ylabel('Number of clusters');
plt.title('DBSCAN clusters by value for eps')

db = DBSCAN(eps=2.21, min_samples=20, random_state=0)
Clusters['DB'] = pd.Series(db.fit_predict(Dataset['t-SNE']), index=Dataset['t-SNE'].index, name='DBSCAN clusters')

Dataset['t-SNE with clusters'] = Dataset['t-SNE'].join(Clusters['DB'])

# Your result
outlier_palette = [(0.8, 0.8, 0.8)];
outlier_palette = outlier_palette + sns.color_palette('spectral', n_colors=9)

sns.lmplot(data=Dataset['t-SNE with clusters'],
           x='Axis 0', y='Axis 1', hue='DBSCAN clusters', palette=outlier_palette,
          size=5, aspect=1, fit_reg=False);

cluster_counts_df = pd.DataFrame()
for cluster_nb in Clusters['DB'].unique():
    cluster_indices = (Clusters['DB'][Clusters['DB']==cluster_nb]).index
    cluster_counts = normalised_value_counts(Dataset['Encoded response'].loc[cluster_indices,'readmitted'])
    cluster_counts = cluster_counts.append(pd.Series(len(cluster_indices), index=['foo'], name='bar'))
    cluster_counts.name = 'Cluster '+str(cluster_nb)
    cluster_counts_df = pd.concat([cluster_counts_df, cluster_counts], axis=1)

priors = normalised_value_counts(Dataset['Encoded response'].loc[Clusters['DB'].index,'readmitted'])
priors.name = 'Priors'
cluster_counts_df = pd.concat([cluster_counts_df, priors], axis=1)
cluster_counts_df.index = ['No', '>30 days', '<30 days', 'Nb. points in cluster']
pd.set_option('max_columns',15)
round(cluster_counts_df,2)

