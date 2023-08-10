import sys, subprocess, collections

# Import Pandas
import pandas as pd

# Import SciKit decision tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Import Scikit cross-validation function
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

# Import naive bayes
from sklearn.naive_bayes import BernoulliNB

# Import module to read in secure data
sys.path.append('../data/NREL')
import retrieve_data as rd

solar = rd.retrieve_dirks_sheet()

sys.path.append('utils')
import process_data as prd

prd.clean_data(solar)

def get_classification_model_data(df, modes):
    """
    Build a DataFrame that holds all of the information for building naive bayes models
    
    Args:
        df (DataFrame): Pandas DataFrame that has been cleaned using the clean_data function
        modes (list of string): List of degradation modes to add as dummy variable columns
    
    Returns:
        DataFrame: Holds the features and target necessary for building naive bayes
    """
    # Create binary dummy data for each of the categorical variables
    naive_df = pd.DataFrame(df.loc[:, 'Mounting'])
    naive_df = naive_df.join(pd.get_dummies(df['Mounting']))
    naive_df = naive_df.join(df.loc[:, 'Climate3'])
    naive_df = naive_df.join(pd.get_dummies(df['Climate3']))

    # Bin the installation year data into binary values (0: <2000, 1: >=2000)
    naive_df = naive_df.join(df.loc[:, 'Begin.Year'])
    naive_df['Begin.Year'].fillna(naive_df['Begin.Year'].mean(), inplace=True)
    bins = [0, 2000, 9999]
    group_names = [0, 1]
    naive_df['After 2000'] = pd.cut(naive_df['Begin.Year'], bins, labels=group_names)

    # Add the cleaned Cause column for visual reference
    naive_df = naive_df.join(df.loc[:, 'Cause (Cleaned)'])

    # Add the dummy variable columns of the degradation modes
    for m in modes:
        naive_df = pd.concat([naive_df, df[m]], axis=1)
    
    return naive_df

def visualize_tree(tree, class_name, feature_names, dot_name, png_name):
    """Create tree png using graphviz
    NOTE: Will export trees into an output folder located on the same level as this notebook

    Args:
        tree -- scikit-learn DecsisionTree
        feature_names -- list of feature names
    """
    with open("output/" + dot_name, 'w') as f:
        export_graphviz(tree, out_file=f,
                        filled=True,
                        class_names=['None', class_name],
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "output/" + dot_name, "-o", "output/" + png_name]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

def generate_decision_tree_models(df, modes, features, params_dict):
    """
    Generate dot and png files for decision tree models for each specified degradation mode
    
    Args:
        df (DataFrame): DataFrame returned by the get_classification_model_data function
        modes (list of string): List of degradation modes to build models for
        features (list of string): List of features (columns in the dataframe) to use for the model
        params_dict (dict): A dictionary containing the min_samples_leaf and max_depth parameters of each degradation mode
                            Format {Key: Degradation mode,Value:{Key:min_samples_leaf,Value:int}
                                                                {Key:max_depth,Value:int}}
    """
    for m in modes:
        # Fitting the decision tree with scikit-learn
        y = df[m]
        x = df.loc[:, feature_names]
        msl = params_dict[m]['min_samples_leaf']
        md = params_dict[m]['max_depth']
        dt = DecisionTreeClassifier(min_samples_split=1000, min_samples_leaf=msl, max_depth=md, random_state=99)
        dt.fit(x, y)
        scores = cross_val_score(dt, x, y, cv=10, scoring='accuracy')
        
        # Draw the decision tree and save the output file
        if m == 'Diode/J-box problem':
            m = 'Diode_J-box problem'
        dot_name = 'Cause ' + m + '.dot'
        png_name = 'Cause ' + m + '.png'
        print('Exported ' + png_name + ' to output/')
        print('Score: ' + str(scores.mean()))
        visualize_tree(dt, m, feature_names, dot_name, png_name)

feature_names = ('Snow', 'Hot & Humid', 'Desert', 'Moderate',
                'roof', 'roof rack', '1-axis tracker', 'rack')
best_params_dict = {'Major delamination': {'max_depth': 2, 'min_samples_leaf': 2000},
                    'Hot spots': {'max_depth': 1, 'min_samples_leaf': 2600},
                    'Encapsulant discoloration': {'max_depth': 1, 'min_samples_leaf': 2800},
                    'Internal circuitry discoloration':{'max_depth': 1, 'min_samples_leaf': 2000},
                    'Fractured cells': {'max_depth': 1, 'min_samples_leaf': 2000},
                    'Glass breakage': {'max_depth': 1, 'min_samples_leaf': 2000},
                    'Permanent soiling': {'max_depth': 1, 'min_samples_leaf': 2000},
                    'Diode/J-box problem': {'max_depth': 1, 'min_samples_leaf': 2000}}
modes = ['Hot spots', 'Encapsulant discoloration', 'Major delamination', 'Internal circuitry discoloration',
         'Fractured cells', 'Glass breakage', 'Permanent soiling', 'Diode/J-box problem']

dt_df = get_classification_model_data(solar, modes)
dt_df

generate_decision_tree_models(dt_df, modes, feature_names, best_params_dict)

def dt_grid_search(df, mode, feature_names, lr, dr):
    """
    Perform grid search on the given DataFrame to find the optimal parameters for each decision tree
    NOTE: Leaf range is searched in a range of 2000 to 3000 at intervals of 100
          Depth range is searched in a range of 1 and 9 at intervals of 1
          
    Args:
        df (DataFrame): DataFrame returned by the get_classification_model_data function
        mode (string): Degradation mode to run grid search for
        feature_names (list of string): List of features (columns in the dataframe) to use for the model
        lr (list of integers or None): Integers to try for the grid search for the min_samples_leaf param
                                       Defaulted to 2000-3000 for every 100
        dr (list of integers or None): Integers to try for the grid search for the max_depth param
                                       Defaulted to 1-9 for every 1
    """
    y = df[m]
    x = df.loc[:, feature_names]
    leaf_range = lr
    depth_range = dr
    if lr == None:
        leaf_range = [a*100 for a in range(20, 31)]
    if dr == None:
        depth_range = [b for b in range(1, 10)]

    dt = DecisionTreeClassifier(min_samples_split=0.01, min_samples_leaf=6000, max_depth=6, random_state=99)
    param_grid = dict(min_samples_leaf=leaf_range, max_depth=depth_range)

    grid = GridSearchCV(dt, param_grid, cv=10, scoring='accuracy')
    grid.fit(x, y)

    print(m)
    print(grid.best_params_)
    print(grid.best_score_)

# Example of how to use the dt_grid_search function
modes = ['Hot spots', 'Encapsulant discoloration', 'Major delamination', 'Internal circuitry discoloration',
     'Fractured cells', 'Glass breakage', 'Permanent soiling', 'Diode/J-box problem']
feature_names = ('Snow', 'Hot & Humid', 'Desert', 'Moderate',
                'roof', 'roof rack', '1-axis tracker', 'rack')    
    
for m in modes:
    dt_grid_search(dt_df, m, feature_names, None, None)

def dt_cross_val_score(df, m, feature_names, msl, md):
    """
    Performs a 10-fold cross validation on a decisition tree built with the specified parameters
    
    Args:
        df (DataFrame): DataFrame returned by the get_classification_model_data function
        m (string): Degradation mode to build the model on
        msl (int): min_samples_leaf parameter for the model
        md (int): max_depth pramater for the model
    Returns:
        float: The average of accuracies from each of the 10 folds
    """
    y = df[m]
    x = df.loc[:, feature_names]
    
    dt = DecisionTreeClassifier(min_samples_split=0.01, min_samples_leaf=msl, max_depth=md, random_state=99)
    scores = cross_val_score(dt, x, y, cv=10, scoring='accuracy')
    return scores.mean()

# Example of how to use the dt_cross_val_score function
feature_names = ('Snow', 'Hot & Humid', 'Desert', 'Moderate',
                'roof', 'roof rack', '1-axis tracker', 'rack')
dt_cross_val_score(dt_df, 'Hot spots', feature_names, 2600, 1)

def generate_naive_bayes_models(df, features, modes):
    """
    Build a dictionary that holds all Bernoulli Naive Bayes models for specified degradation modes
    NOTE: This function will print the score provided by Scikit of each model
    
    Args:
        df (DataFrame): DataFrame returned by the get_classification_model_data function
        features (list of string): List of desired columns to include in the model as features
        modes (list of string): List of degradation modes to build models for
    Returns:
        dict: Dictionary to hold all Bernoulli Naive Bayes models
              Format is {Key:Degradation mode, Value:respective Naive Bayes model}
    """
    nb_dict = {}

    # Columns to fit against the target for the Naive Bayes models
    X = df.loc[:, features]

    # Find score of Bernoulli Naive Bayes models for each degradation mode
    for m in modes:
        y = df[m]
        clf = BernoulliNB()
        clf.fit(X, y)
        print('Cause: ' + m)
        print('Score: ' + str(clf.score(X, y)))
        nb_dict[m] = clf
        
    return nb_dict

modes = ['Hot spots', 'Encapsulant discoloration', 'Major delamination', 'Internal circuitry discoloration',
         'Fractured cells', 'Glass breakage', 'Permanent soiling', 'Diode/J-box problem']

nb_df = get_classification_model_data(solar, modes)
nb_df.head()

features = ['1-axis tracker', '2-axis tracker', 'rack', 'roof', 'roof rack',
                     'single-axis', 'Desert', 'Hot & Humid', 'Moderate', 'Snow', 'After 2000']

nb_dict = generate_naive_bayes_models(nb_df, features, modes)
nb_dict

# Scikit score of individual naive bayes model
X = nb_df.loc[:, features]
nb_dict['Hot spots'].score(X,nb_df['Hot spots'])

# Posterior probability of hot spots with 1-axis tracker and Desert and BEFORE 2000
nb_dict['Hot spots'].predict_proba([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

# Posterior probability of hot spots with 1-axis tracker and Desert and AFTER 2000
nb_dict['Hot spots'].predict_proba([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]])

