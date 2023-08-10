import pandas as pd
import numpy as np
from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")
datasets = {
	"student" : {
		"train_name" : "prep_data/student/student_grades.csv",
		"X_col" : range(33),
		"Y_col" : [33],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"contraceptive" : {
		"train_name" : "prep_data/contraceptive/contraceptive.csv",
		"X_col" : range(9),
		"Y_col" : [9],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"autism" : {
		"train_name" : "prep_data/Autism-Adult-Data/Autism-Adult-Data-preproc.csv",
		"X_col" : range(20),
		"Y_col" : [20],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"bankruptcy" : {
		"train_name" : "prep_data/bankruptcy/bankrupt.csv",
		"X_col" : range(6),
		"Y_col" : [6],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"breast_cancer" : {
		"train_name" : "prep_data/breast-cancer/breast-cancer-wisconsin.data",
		"X_col" : range(9),
		"Y_col" : [9],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"horse" : {
		"train_name" : "prep_data/horse-colic/horse-colic.data-preproc.csv",
		"X_col" : range(22),
		"Y_col" : [22],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
    "hr" : {
		"train_name" : "prep_data/hr-analytics/HR_comma_sep.csv",
		"X_col" : range(9),
		"Y_col" : [9],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"english" : {
		"train_name" : "prep_data/teaching-english/tae.csv",
		"X_col" : range(5),
		"Y_col" : [5],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"phishing" : {
		"train_name" : "prep_data/website-phishing/PhishingData.csv",
		"X_col" : range(9),
		"Y_col" : [9],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"wine" : {
		"train_name" : "prep_data/wine-quality/winequality-red.csv",
		"X_col" : range(11),
		"Y_col" : [11],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"amazon" : {
		"train_name" : "prep_data/amazon/amzreviews.csv",
		"X_col" : range(1,3093),
		"Y_col" : [3093],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"congress" : {
		"train_name" : "prep_data/congress/congress_leave.csv",
		"X_col" : range(1,17),
		"Y_col" : [17],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"covertype" : {
		"train_name" : "prep_data/covertypes/covertype_scale.csv",
		"X_col" : range(54),
		"Y_col" : [54],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	},
	"kidney" : {
		"train_name" : "prep_data/kidney/kidney_colMeanMode.csv",
		"X_col" : range(24),
		"Y_col" : [24],
		"has_header" : True,
		"filetype" : "CSV",
		"encode_labels" : False
	}
}

def read_dataset(dataset):
    df = pd.read_csv('../'+dataset["train_name"])
    data_X = df.iloc[:, dataset["X_col"]].copy()
    data_y = df.iloc[:, dataset["Y_col"]].copy()
    assert(data_y.columns[0] == 'Class')
    return data_X, data_y

header = "model,dataset,scoring,p1,p2,acc"
output = "../results/grid_search_res.csv"
#with open(output, "a") as f:
#    f.write(header+'\n')
parameters = {'svm': {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [1,10,100,1000]},
              'rf': {'n_estimators': [10, 30, 50, 100], 'max_depth': [None,1,3]},
              'knn': {'n_neighbors': [1,3,5,8,10,30,50,100], 'weights': ['uniform', 'distance']},
              'mnb': {},
              'mlp': {'activation': ['identity', 'relu', 'logistic'], 'alpha': [0.001, 1.0000000000000001e-05, 9.9999999999999995e-07]}}

# Split the dataset 70/30
total_iter = 14*4*3
current_iter = 0
for dataset in list(datasets.keys()):
    print("Tuning parameters for %s dataset" % dataset)
    X, y = read_dataset(datasets[dataset])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    scores = ['accuracy','precision_macro', 'recall_macro']
    for label, model in {'svm': SVC(),'knn':KNeighborsClassifier(),'rf': RandomForestClassifier(), 'mlp': MLPClassifier()}.items():
        for score in scores:
            current_iter += 1
            print("[%s]%d/%d" % (dataset,current_iter, total_iter))
            #print("# Tuning hyper-parameters for %s, model %s" % (score, model.__class__.__name__))
            #print()
            clf = GridSearchCV(model, parameters[label], cv=5,scoring=('%s' % score))
            clf.fit(X_train, y_train)
            
            #print("Best parameters set found on development set:")
            #print()
            #print(clf.best_params_)
            #print()
            #print("Grid scores on development set:")
            #print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            #    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            #print()

            #print("Detailed classification report:")
            #print()
            #print("The model is trained on the full development set.")
            #print("The scores are computed on the full evaluation set.")
            #print()
            y_true, y_pred = y_test, clf.predict(X_test)
            #print(classification_report(y_true, y_pred))
            #print()
            res_str = "%s,%s,%s,%s,%s,%.2f" % (model.__class__.__name__,
                                                    dataset,
                                                    score,
                                                    list(clf.best_params_.values())[0],
                                                    list(clf.best_params_.values())[1],
                                                    (max(means)+accuracy_score(y_true, y_pred))/2)
                                                        
            #print(res_str)
            with open(output, "a") as f:
                f.write(res_str + '\n')

import pandas as pd
import numpy as np
from IPython.display import display, Markdown, Latex, HTML



df_def = pd.read_csv('../results/grid_search_res.csv')
res = df_def.drop(['dataset', 'scoring'], axis=1).groupby(['model','p1','p2']).count()
display(HTML(res.to_html()))



