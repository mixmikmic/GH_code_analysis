#data handling/prediction
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.externals.six import StringIO
from sklearn import metrics
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

#visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz

# read data into a DataFrame
data = pd.read_csv("../data/data.csv")
print(data.dtypes)
print(data.shape)
data.head()

clean_data = data
print("clean_data rows/cols before cleanup: ",clean_data.shape)

# Drop columns we don't need for modeling
del clean_data["Name"]
del clean_data["AnimalID"]
del clean_data["DateTime"]
del clean_data["OutcomeSubtype"]

# Convert outcome to 1 = Adoption, 0 = everything else
print(clean_data.OutcomeType.value_counts())
# clean_data.loc[:,('OutcomeType')] = (clean_data.OutcomeType=='Adoption').astype(int)
clean_data['OutcomeType'] = (clean_data.OutcomeType=='Adoption').astype(int)
print(clean_data.OutcomeType.value_counts())

# Inspect columns with missing values
print("Columns with missing values:")
print(clean_data.isnull().any())
# Check for # of missing values per column
print(clean_data.isnull().sum())

# Delete samples with missing values
clean_data_nonnull = clean_data.dropna()
print("Fraction of data kept:\n",float(clean_data_nonnull.shape[0])/clean_data.shape[0])
clean_data = clean_data_nonnull

def convert_age_to_days(age):
    a,unit = age.split(" ")
    a = int(a)
    num_of_days = 0
    mult = 1
    if unit == "years" or unit == "year":
        mult = 365
    elif unit == "months" or unit == "month":
        mult = 30
    elif unit == "weeks" or unit == "week":
        mult = 7
        
    num_of_days = a * mult
    return num_of_days

# Remove 22 rows with "0 years" as age
clean_data = clean_data[clean_data.AgeuponOutcome != "0 years"]

# Convert age from string to number of days
clean_data.loc[:,['AgeuponOutcome']] = clean_data["AgeuponOutcome"].apply(convert_age_to_days)
# clean_data.AgeuponOutcome.value_counts().sort_values(ascending=False)

clean_age_data = clean_data.AgeuponOutcome.values.reshape(-1, 1)

# Scale clean_data.AgeuponOutcome
scaler = StandardScaler() #create a scaler object
scaler.fit_transform(clean_age_data) #fit the scaler
clean_age_data_scaled = scaler.transform(clean_age_data) #transform the data with it

# compare original to standardized
# print("original values:\n",clean_age_data[:10],"\n")
# print("scaled values:\n",clean_age_data_scaled[:10],"\n")

# figure out how the standardization worked
print("Mean of column:\n",scaler.mean_,"\n")
print("standard deviation of column:\n",scaler.scale_,"\n")
print("Final Means of scaled data:\n",clean_age_data_scaled.mean(axis=0))
print("Final standard deviation of scaled data:\n",clean_age_data_scaled.std(axis=0))

clean_data.loc[:,("AgeuponOutcome")] = clean_age_data_scaled

print("Age min value: ",clean_data.AgeuponOutcome.min())
print("Age min value: ",clean_data.AgeuponOutcome.max())

# Extract sex from the same column that tells us whether animal was neutered, spayed, or left intact
def get_sex(val):
    if val.find('Male') >= 0: return 'Male'
    if val.find('Female') >= 0: return 'Female'
    return 'UnknownSex'

def get_neutered(val):
    if val.find('Spayed') >= 0: return 'Spayed'
    if val.find('Neutered') >= 0: return 'Neutered'
    if val.find('Intact') >= 0: return 'Intact'
    return 'UnknownNeuteredSpayed'

print("Before extraction:")
print(clean_data.SexuponOutcome.value_counts())
clean_data.loc[:,("Sex")] = clean_data.SexuponOutcome.apply(get_sex)
clean_data.loc[:,("NeuteredSpayed")] = clean_data.SexuponOutcome.apply(get_neutered)
print("After extraction: ")
print(clean_data.Sex.value_counts())

# One-hot code AnimalType - Dog or Cat
animal_type_one_coded = clean_data.AnimalType.str.get_dummies()

# One-hot code Sex
sex_upon_outcome_one_coded = clean_data.Sex.str.get_dummies()

# One-hot code Neutered, Spayed
neutered_spayed_one_coded = clean_data.NeuteredSpayed.str.get_dummies()

# One-hot code Color
# print(clean_data.Color.value_counts())
popular_colors = clean_data.Color.value_counts(ascending=False).head(5)
print("Top 5 colors: ",popular_colors)

def color_bucket(color):
    if color in popular_colors:
        return color
    
    return "OtherColor"

clean_data.loc[:,("Color")] = clean_data["Color"].apply(color_bucket)
color_one_coded = clean_data.Color.str.get_dummies()

# One-hot code Breed
popular_breeds = clean_data.Breed.value_counts(ascending=False).head(5)
print("Top 5 breeds: ",popular_breeds)

def breed_bucket(breed):
    if breed in popular_breeds:
        return breed
    
    return "OtherBreed"

clean_data.loc[:,("Breed")] = clean_data["Breed"].apply(breed_bucket)
breed_one_coded = clean_data.Breed.str.get_dummies()

# Merge one-hot coded dataframes with the original df
clean_data = clean_data.merge(animal_type_one_coded,left_index=True,right_index=True).merge(sex_upon_outcome_one_coded,left_index=True,right_index=True).merge(breed_one_coded,left_index=True,right_index=True).merge(color_one_coded,left_index=True,right_index=True).merge(neutered_spayed_one_coded,left_index=True,right_index=True)

# Drop original columns that aren't needed anymore
del clean_data["Breed"]
del clean_data["AnimalType"]
del clean_data["SexuponOutcome"]
del clean_data["Color"]
del clean_data["Sex"]
del clean_data["NeuteredSpayed"]

clean_data.columns = clean_data.columns.map(lambda x: x.replace(" ",""))
clean_data.columns = clean_data.columns.map(lambda x: x.replace("/",""))

clean_data.head()

# Split data into (adopted vs not) and plot correlation matrix heatmaps 

adopted_data = pd.DataFrame(clean_data[clean_data.OutcomeType == 1])
not_adopted_data = pd.DataFrame(clean_data[clean_data.OutcomeType == 0])

adopted_data.drop(['OutcomeType'],inplace=True, axis=1)
not_adopted_data.drop(['OutcomeType'],inplace=True, axis=1)

sns.mpl.pyplot.figure(figsize=(15, 12))
sns.heatmap(adopted_data.corr(),annot=True, fmt=".1f");

sns.mpl.pyplot.figure(figsize=(15, 12))
sns.heatmap(not_adopted_data.corr(),annot=True, fmt=".1f");

# Examine split of adopted vs not
print(clean_data.OutcomeType.value_counts())
print(clean_data.OutcomeType.value_counts()/clean_data.OutcomeType.value_counts().sum())

sns.countplot(clean_data.OutcomeType);

selected_features = clean_data.columns.tolist()[1:]
all_features_X = clean_data[selected_features]
all_y = clean_data.OutcomeType

all_X_train,all_X_test,all_y_train,all_y_test = train_test_split(all_features_X,all_y,random_state=42)

logreg_all = LogisticRegression(C=1e9)
logreg_all.fit(all_X_train,all_y_train)

dumb_model = DummyClassifier(strategy='most_frequent')
dumb_model.fit(all_X_train, all_y_train)
y_dumb_class = dumb_model.predict(all_X_test)

print("Bias coefficient (intercept): {0}".format(logreg_all.intercept_[0]))
print("Model accuracy on {0}% training data: {1}".format(int((1-n) * 100),metrics.accuracy_score(all_y_train,logreg_all.predict(all_X_train))))
all_y_test_pred = logreg_all.predict(all_X_test)
print("Model accuracy on {0}% test data: {1}".format(int(n * 100),metrics.accuracy_score(all_y_test,all_y_test_pred)))
print("Most frequent class dummy classifier test accuracy: ",metrics.accuracy_score(all_y_test, y_dumb_class))
print("F1 Score:\n", metrics.f1_score(all_y_test,all_y_test_pred))
print("\r") 
print("Features in order of their absolute coefficient values--indicating their importance")
feature_coeff_df = pd.DataFrame(list(zip(clean_data.columns.tolist()[1:],logreg_all.coef_[0],abs(logreg_all.coef_[0]))),columns=["feature","coefficient","abs_coefficient"])
print(feature_coeff_df.sort_values(['abs_coefficient'],ascending=False))

print("\n")

X_train,X_test,y_train,y_test = train_test_split(all_features_X,all_y,random_state=1)

depths = range(1,30)
train_accuracy, test_accuracy = [],[]
for depth in depths:
    decision_tree = DecisionTreeClassifier(max_depth=depth,random_state=10)
    decision_tree.fit(X_train,y_train)
    curr_train_accuracy = metrics.accuracy_score(y_train,decision_tree.predict(X_train))
    y_test_pred = decision_tree.predict(X_test)
    curr_test_accuracy = metrics.accuracy_score(y_test,y_test_pred)
    train_accuracy.append(curr_train_accuracy)
    test_accuracy.append(curr_test_accuracy)
#     print("F1 Score:\n", metrics.f1_score(y_test,y_test_pred))
    
sns.mpl.pyplot.plot(depths,train_accuracy,label='train_accuracy')
sns.mpl.pyplot.plot(depths,test_accuracy,label='test_accuracy')
sns.mpl.pyplot.xlabel("maximum tree depth")
sns.mpl.pyplot.ylabel("accuracy score")
sns.mpl.pyplot.legend();

#DecisionTreeClassifier based on the best max_depth = 6 observed above
best_single_tree = DecisionTreeClassifier(max_depth=6, random_state=1)
best_single_tree.fit(X_train,y_train)
best_single_tree_y_pred = best_single_tree.predict(X_test)
print("Best Decision Tree Accuracy Score:",metrics.accuracy_score(y_test,best_single_tree_y_pred))

dumb_model = DummyClassifier(strategy='most_frequent')
dumb_model.fit(X_train, y_train)
y_dumb_class = dumb_model.predict(X_test)
print("Most frequent class dummy classifier test accuracy: ",metrics.accuracy_score(y_test, y_dumb_class))
print("F1 Score: ", metrics.f1_score(y_test,best_single_tree_y_pred))

#Plot decision tree
dot_data = StringIO()  
export_graphviz(best_single_tree, out_file=dot_data,  
                    feature_names=X_train.columns.tolist(),  
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  

#Compute the feature importances
feature_imp = pd.DataFrame({'feature':X_train.columns.tolist(), 'importance':best_single_tree.feature_importances_})
print(feature_imp.sort_values("importance",ascending=False))

rf = RandomForestClassifier(n_estimators=50) #random forest with 50 trees

# Compute cross-validation score accuracy across 10 folds
cross_val_scores = cross_val_score(rf,all_features_X,all_y,cv=10)
print("10-fold accuracies:\n",cross_val_scores)
print("Mean cv-accuracy:",np.mean(cross_val_scores))
print("Std of cv-accuracy:",np.std(cross_val_scores))

# Use GridSearchCV to automate the search to find best max_depth per-tree across depths 1-10
rf_grid = RandomForestClassifier(n_estimators=50,random_state=1,n_jobs=-1) #50 trees
max_depth_range = range(1, 20)
param_grid = dict(max_depth=max_depth_range)
grid = GridSearchCV(rf_grid, param_grid, cv=10, scoring='accuracy')
grid.fit(all_features_X, all_y)

# Examine the results of the grid search
# grid.cv_results_

grid_mean_scores = grid.cv_results_["mean_test_score"]

# Plot the results
sns.mpl.pyplot.plot(max_depth_range, grid_mean_scores)
sns.mpl.pyplot.xlabel('max_depth')
sns.mpl.pyplot.ylabel('Cross-Validated Mean Test Set Accuracy')

# Examine the best model
print("Best score:",grid.best_score_)
print("Best params:",grid.best_params_)
print("Best estimator:",grid.best_estimator_)

# Use GridSearchCV to automate the search for multiple parameters 
# (max_depth and min_samples_leaf) across 50 trees with depths range 1-10 and leaf samples range 1-8
rf_grid = RandomForestClassifier(n_estimators=50,random_state=1,n_jobs=-1) 
max_depth_range = range(1, 20)
leaf_range = range(1, 10)
param_grid2 = dict(max_depth=max_depth_range, min_samples_leaf=leaf_range)
grid2 = GridSearchCV(rf_grid, param_grid2, cv=10, scoring='accuracy')
grid2.fit(all_features_X, all_y)

print("Best GridSearchCV score:",grid2.best_score_)
print("Best GridSearchCV params:",grid2.best_params_)

print("Top 5 features based on importance generated by LogisticRegression:",
      feature_coeff_df.sort_values(['abs_coefficient'],ascending=False).head(5))

print("Top 5 features based on importance generated by DecisionTreeClassifier:",
      feature_imp.sort_values("importance",ascending=False).head(5))

selected_features_logit = feature_coeff_df.sort_values(['abs_coefficient'],ascending=False).head(5).loc[:,'feature'].tolist()
selected_features_dtree = feature_imp.sort_values("importance",ascending=False).head(5).loc[:,'feature'].tolist()

select_features_logit_X = clean_data[selected_features_logit]
select_features_dtree_X = clean_data[selected_features_dtree]

all_y = clean_data.OutcomeType

print("Correlation matrix for top 5 features based on Logit")
sns.heatmap(select_features_logit_X.corr(),annot=True, fmt=".1f");

print("Correlation matrix for top 5 features based on DTree")
sns.heatmap(select_features_dtree_X.corr(),annot=True, fmt=".1f");

def logistic_regression(features_X,features_list,features_source):
    for n in [0.3,0.2,0.1]:
        all_X_train,all_X_test,all_y_train,all_y_test = train_test_split(features_X,all_y,test_size=n,random_state=1)

        logreg_all = LogisticRegression(C=1e9)
        logreg_all.fit(all_X_train,all_y_train)

        print("========== Data split {0}/{1} with select features based on {2}".format(int((1-n) * 100),int(n * 100),features_source))
        print("Bias coefficient (intercept): {0}".format(logreg_all.intercept_[0]))
        print("Model accuracy on {0}% training data: {1}".format(int((1-n) * 100),metrics.accuracy_score(all_y_train,logreg_all.predict(all_X_train))))
        print("Model accuracy on {0}% test data: {1}".format(int(n * 100),metrics.accuracy_score(all_y_test,logreg_all.predict(all_X_test))))
        print("\r") 
        print("Features in order of their absolute coefficient values--indicating their importance")
        feature_coeff_df = pd.DataFrame(list(zip(features_list,logreg_all.coef_[0],abs(logreg_all.coef_[0]))),columns=["feature","coefficient","abs_coefficient"])
        print(feature_coeff_df.sort_values(['abs_coefficient'],ascending=False))
        print("\n")
        
print("########## Accuracy based on select features from Logit ##########")
logistic_regression(select_features_logit_X,selected_features_logit,"Logit")
print("########## Accuracy based on select Features from Dtree ##########")
logistic_regression(select_features_dtree_X,selected_features_dtree,"DTree")

def decision_tree(features_X,features_source):
    X_train,X_test,y_train,y_test = train_test_split(features_X,all_y,test_size=.3,random_state=123)

    depths = range(1,100)
    train_accuracy, test_accuracy = [],[]
    for depth in depths:
        decision_tree = DecisionTreeClassifier(max_depth=depth,random_state=10)
        decision_tree.fit(X_train,y_train)
        curr_train_accuracy = metrics.accuracy_score(y_train,decision_tree.predict(X_train))
        curr_test_accuracy = metrics.accuracy_score(y_test,decision_tree.predict(X_test))
#         print(features_source + " -- decision Tree Train/Test accuracy for depth "+str(depth),curr_train_accuracy," ",curr_test_accuracy)
        train_accuracy.append(curr_train_accuracy)
        test_accuracy.append(curr_test_accuracy)
    sns.mpl.pyplot.plot(depths,train_accuracy,label=features_source+'_train_accuracy')
    sns.mpl.pyplot.plot(depths,test_accuracy,label=features_source+'_test_accuracy')
    sns.mpl.pyplot.xlabel("maximum tree depth")
    sns.mpl.pyplot.ylabel("accuracy score")
    sns.mpl.pyplot.legend();

decision_tree(select_features_logit_X,"logit_feat")
decision_tree(select_features_dtree_X,"dtree_feat")

#DecisionTreeClassifier based on the best max_depth = 8 observed from 10 trials
best_single_tree = DecisionTreeClassifier(max_depth=8, random_state=1)
best_single_tree.fit(X_train,y_train)
print("Best Decision Tree Accuracy Score:",metrics.accuracy_score(y_test,best_single_tree.predict(X_test)))

#Plot decision tree
dot_data = StringIO()  
export_graphviz(best_single_tree, out_file=dot_data,  
                    feature_names=X_train.columns.tolist(),  
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  

def random_forest(features_X):    
    rf = RandomForestClassifier(n_estimators=50) #random forest with 50 trees

    # Compute cross-validation score accuracy across 10 folds
    cross_val_scores = cross_val_score(rf,features_X,all_y,cv=10)
    print("10-fold accuracies:\n",cross_val_scores)
    print("Mean cv-accuracy:",np.mean(cross_val_scores))
    print("Std of cv-accuracy:",np.std(cross_val_scores))
    
print("########## Accuracy based on select features from Logit ##########")
random_forest(select_features_logit_X)
print("########## Accuracy based on select Features from Dtree ##########")
random_forest(select_features_dtree_X)

def grid_search_cv_max_depth(features_X):
    # Use GridSearchCV to automate the search to find best max_depth per-tree across depths 1-10
    rf_grid = RandomForestClassifier(n_estimators=50,random_state=1,n_jobs=-1) #50 trees
    max_depth_range = range(5, 20)
    param_grid = dict(max_depth=max_depth_range)
    grid = GridSearchCV(rf_grid, param_grid, cv=10, scoring='accuracy')
    grid.fit(features_X, all_y)

    # Examine the results of the grid search
    # grid.cv_results_

    grid_mean_scores = grid.cv_results_["mean_test_score"]

    # Plot the results
#     sns.mpl.pyplot.plot(max_depth_range, grid_mean_scores)
#     sns.mpl.pyplot.xlabel('max_depth')
#     sns.mpl.pyplot.ylabel('Cross-Validated Mean Test Set Accuracy')

    # Examine the best model
    print("Best score:",grid.best_score_)
    print("Best params:",grid.best_params_)
    print("Best estimator:",grid.best_estimator_)

print("########## Best params based on select features from Logit ##########")
grid_search_cv_max_depth(select_features_logit_X)
print("########## Best params based on select Features from Dtree ##########")
grid_search_cv_max_depth(select_features_dtree_X)

def grid_search_cv_maxdepth_minsamples(features_X):
    # Use GridSearchCV to automate the search for multiple parameters 
    # (max_depth and min_samples_leaf) across 50 trees with depths range 1-10 and leaf samples range 1-8
    rf_grid = RandomForestClassifier(n_estimators=50,random_state=1,n_jobs=-1) 
    max_depth_range = range(5, 20)
    leaf_range = range(1, 8)
    param_grid2 = dict(max_depth=max_depth_range, min_samples_leaf=leaf_range)
    grid2 = GridSearchCV(rf_grid, param_grid2, cv=10, scoring='accuracy')
    grid2.fit(features_X, all_y)

    print("Best GridSearchCV score:",grid2.best_score_)
    print("Best GridSearchCV params:",grid2.best_params_)
    
print("########## Best params based on select features from Logit ##########")
grid_search_cv_maxdepth_minsamples(select_features_logit_X)
print("########## Best params based on select Features from Dtree ##########")
grid_search_cv_maxdepth_minsamples(select_features_dtree_X)



