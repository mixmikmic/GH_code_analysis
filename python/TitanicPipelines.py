get_ipython().magic('matplotlib inline')

import pandas
import seaborn
seaborn.set_style("whitegrid")
seaborn.set_palette(seaborn.color_palette("GnBu_d"))
import numpy

import sklearn
from sklearn import *

trainData = pandas.read_csv("train.csv", index_col=0)
testData = pandas.read_csv("test.csv", index_col=0)

responseVariable = "Survived"

trainData.head()

testData.head()

"""
VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)

SPECIAL NOTES:
Pclass is a proxy for socio-economic status (SES)
 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

Age is in Years; Fractional if Age less than One (1)
 If the Age is Estimated, it is in the form xx.5

With respect to the family relation variables (i.e. sibsp and parch)
some relations were ignored.  The following are the definitions used
for sibsp and parch.

Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
Parent:   Mother or Father of Passenger Aboard Titanic
Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

Other family relatives excluded from this study include cousins,
nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
only with a nanny, therefore parch=0 for them.  As well, some
travelled with very close friends or neighbors in a village, however,
the definitions do not support such relations.
""";

XTrain = trainData.ix[:, trainData.columns.difference([responseVariable])]
yTrain = trainData.ix[:, responseVariable]
XTest = testData

class Categorize(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Converts given columns into pandas dtype 'category'.
    """
    
    def __init__(self, columns, encode=True):
        self.columns = columns
        self.encode = encode
    
    def fit(self, X, y):
        return self
        
    
    def transform(self, X):
        #print("categorizing columns: {0}".format(self.columns))
        for column in self.columns:
            X[column] = X[column].astype("category")
            if self.encode:
                X[column] = X[column].cat.codes
        return X

class AnnotateMissingValues(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    For any feature that has missing values, create an extra boolean
    feature indicating whether the item has a missing values.
    """
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        nullTable = X.isnull().any()
        self.nullColumns = [column for (column, isNull) in nullTable.iteritems() if isNull]
        #print("annotating missing values for columns: {0}".format(self.nullColumns))
        for column in self.nullColumns:
            X["{0}_Missing".format(column)] = X[column].isnull()
        X = X[sorted(list(X.columns))]
        return X
    
    

class DropColumns(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Drops all given columns from the data frame.
    """
    
    def __init__(self, columns=[]):
        self.columns = columns
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        #print("dropping columns: {0}".format(self.columns))
        X = X.drop(self.columns, axis=1)
        return X

class LabelEncodeColumns(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Applies sklearn.preprocessing.LabelEncoder to multiple columns
    """
    
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y):
        return self
  
    def transform(self, X):
        X[self.columns] = X[self.columns].apply(sklearn.preprocessing.LabelEncoder().fit_transform)
        return X

class StandardScaleColumns(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    TODO: work in progress
    Applies sklearn.preprocessing.StandardScaler to multiple columns of the DataFrame
    """
    
    def __init__(self, columns, dropOriginal=True):
        self.columns = columns
        self.dropOriginal = dropOriginal # drop the original columns after transformation
        
    def fit(self, X, y):
        toTransform, rest = X[self.columns],  X[X.columns.difference(self.columns)]
        toTransformArray = toTransform.as_matrix()
        self.scaler = sklearn.preprocessing.StandardScaler().fit(toTransformArray, y)
        return self
  
    def transform(self, X):
        toTransform, rest = X[self.columns],  X[X.columns.difference(self.columns)]
        transformedArray = self.scaler.transform(X.as_matrix())
        transformed = pandas.DataFrame(transformedArray, columns=columns)
        
    

class AsTransformer(sklearn.base.BaseEstimator):
    """
    Wrap any function in a transformer class.
    """

    def __init__(self, func):
        self._func = func

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, *args, **kwargs):
        return self._func(X, *args, **kwargs)

def extractTitles(X):
    X["Title"] = X["Name"].apply(lambda name: name.split(", ")[1].split(" ")[0])
    return X

allColumns = XTrain.columns

categorial = ["Sex", 
              "Pclass",
              "Ticket", # ? can features be extracted from ticket number?
              "Embarked",
              "Cabin"
             ]

categorialEngineered = ["Title"]

numeric = ["Age",
          "SibSp",
          "Parch",
          "Fare"]

from sklearn.pipeline import Pipeline

preprocessingPipelines = {
    "numeric, imputed": Pipeline(steps = [
                                        ("drop all non-numeric", DropColumns(allColumns.difference(numeric))),
                                        ("impute missing values", sklearn.preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
                                        ]
                                ),
    "numeric, annotated": Pipeline(steps = [
                                        ("drop all non-numeric", DropColumns(allColumns.difference(numeric))),
                                        ("annotate missing values", AnnotateMissingValues()),
                                        ("impute missing values", sklearn.preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
                                        ]
                                ),
    "categorial, labelled": Pipeline(steps = [
                                        ("drop name", DropColumns(["Name"])),
                                        ("annotate missing values", AnnotateMissingValues()),
                                        ("strings to category labels",  Categorize(categorial)),
                                        ("impute missing values", sklearn.preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
                                        ]
                                ),
    "with title feature": Pipeline(steps = [
                                    ("engineer title feature", AsTransformer(extractTitles)),
                                    ("drop name", DropColumns(["Name"])),
                                    ("annotate missing values", AnnotateMissingValues()),
                                    ("strings to category labels",  Categorize(categorial + ["Title"])),
                                    ("impute missing values", sklearn.preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
                                    ]
                            ),
}

preprocessingPipelines["with title feature"].fit(XTrain, yTrain).transform(XTrain)

class BestByCV(sklearn.base.BaseEstimator):
    """
    Estimator which fits given collection of estimators to the data
    and then uses the one for prediction which performs best in a cross-validation.
    """
    
    
    def __init__(self, estimators, scoring, greaterIsBetter):
        self.estimators = estimators # dict: estimator name -> estimator instance
        self.scorer = sklearn.metrics.make_scorer(scoring, greater_is_better=greaterIsBetter)
        self.greaterIsBetter = greaterIsBetter
    
    def fit(self, X, y):
        # fit estimators to training set
        for (estimatorName, estimator) in self.estimators.items():
            estimator.fit(X,y)
        # run cross-validation to determine scores
        self.scores = {}
        self.meanScore = {}
        for(estimatorName, estimator) in self.estimators.items():
            self.scores[estimatorName] = model_selection.cross_val_score(estimator, X, y, scoring=self.scorer)
            self.meanScore[estimatorName] = numpy.mean(self.scores[estimatorName])
        print("mean score for models: ", self.meanScore)
        # select estimator with best score
        bestEstimatorName = max(self.meanScore, key=self.meanScore.get)
        self.bestEstimator = self.estimators[bestEstimatorName]
        print("best estimator : {0}".format(bestEstimatorName))
        return self
    
    def getRankingTable(self):
        cvScoreData = pandas.DataFrame(list(self.meanScore.items()), columns=["estimator", "score"]).sort_values(by="score", ascending=not self.greaterIsBetter)
        cvScoreData.index = range(1,len(self.meanScore) + 1)
        return cvScoreData
    
    def plotRanking(self):
        import matplotlib.pyplot as plt
        cvScores = self.getRankingTable()
        plt.figure(figsize=(14,2))
        plt.bar(cvScores.index, cvScores["score"])
        plt.xticks(cvScores.index, cvScores["estimator"], rotation=25)
        plt.suptitle("Cross Validation Scores")
        
    
    def predict(self, X):
        return self.bestEstimator.predict(X)
        

classifiers = {
                "Decision Tree": (tree.DecisionTreeClassifier, {}),
                "Random Forest": (ensemble.RandomForestClassifier, {}),
                "Naive Bayes": (naive_bayes.GaussianNB, {}),
                "Support Vector Machine": (svm.SVC, {}),
                "Nearest Neighbors" : (neighbors.KNeighborsClassifier, {}),
                "Gradient Boosting (sklearn)": (ensemble.GradientBoostingClassifier, {"random_state": 42}),
                "Logistic Regression" : (linear_model.logistic.LogisticRegression, {}),
                "Multilayer Perceptron" : (neural_network.MLPClassifier, {})
             }

def instantiate(classesAndParams):
    instances = {}
    for (name, (Class, params)) in classesAndParams.items():
        instances[name] = Class(**params)
    return instances
    

prepro = preprocessingPipelines["with title feature"]
model = BestByCV(instantiate(classifiers), scoring=sklearn.metrics.accuracy_score, greaterIsBetter=True)
XTrainPre = prepro.fit_transform(XTrain, yTrain)
prediction = model.fit(XTrainPre, yTrain).predict(XTrainPre)

model.getRankingTable()

model.plotRanking()

BestByCV()

prepro = preprocessingPipelines["categorial, labelled"]
prepro.fit(XTrain, yTrain)

XTrainPre = prepro.transform(XTrain)
XTestPre = prepro.transform(XTest)

predictor = BestByCV(instantiate(classifiers), scoring=sklearn.metrics.accuracy_score, greaterIsBetter=True)

predictor.fit(XTrainPre, yTrain)

predictor.plotRanking()

predictions = predictor.predict(XTestPre)

print("survival rate: {0}".format((predictions == 1).sum() / predictions.shape[0]))

submission = pandas.DataFrame({"PassengerId": testData.index, "Survived": pandas.Series(predictions)})

submission.head()

submission.to_csv("submission.csv", index=False)



