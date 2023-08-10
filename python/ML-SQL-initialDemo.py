#Libraries
#from pyparsing import Word, Literal, alphas, Optional, OneOrMore, Group, Or, Combine, oneOf
from pyparsing import *
import string
import sys
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

letters = string.ascii_letters
punctuation = string.punctuation
numbers = string.digits
whitespace = string.whitespace

#combinations
everything = letters + punctuation + numbers
everythingWOQuotes = everything.replace("\"", "").replace("'", "")

#Booleans
bools = Literal("True") + Literal("False")

#Parenthesis and Quotes
openParen = Literal("(").suppress()
closeParen = Literal(")").suppress()
Quote = Literal('"').suppress()

#includes every combination except whitespace
everything

filename = Word(everything).setResultsName("filename")

#define so that there can be multiple verisions of READ
readKeyword = oneOf(["Read", "READ"]).suppress()

#Define Read Optionals
#header
headerLiteral = (Literal("header") + Literal("=")).suppress()
header = Optional(headerLiteral + Or(bools).setResultsName("header"), default = "False" )

#separator
separatorLiteral = (Or([Literal("sep"), Literal("separator")]) + Literal("=")).suppress()
definesep = Quote + Word(everythingWOQuotes + whitespace).setResultsName("sep") + Quote
separator = Optional(separatorLiteral + definesep, default = ",")

#Compose Read Optionals
readOptions = Optional(openParen + separator + header + closeParen)

read = readKeyword + filename + readOptions

readTest = 'READ /home/ubuntu/notebooks/ML-SQL/Classification/iris.data (sep="," header=False)'

readTestResult = read.parseString(readTest)

filename = readTestResult.filename
header = readTestResult.header
print(header)
sep = readTestResult.sep
print(sep)

#Function to lower a string value of "True" or "False" to an actual python boolean value
def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return None
    else:
         raise ValueError ("Cannot lower value " + s + " to a boolean value")
            
#read parameters from parsed statement and read the file
f = pd.read_csv(filename, sep = sep, header = str_to_bool(header))
f.head()



#define so that there can be multiple verisions of Split
splitKeyword = oneOf(["Split", "SPLIT"]).suppress()

#Phrases used to organize splits
trainPhrase = (Literal("train") + Literal("=")).suppress()
testPhrase = (Literal("test") + Literal("=")).suppress()
valPhrase = (Literal("validation") + Literal("=")).suppress()

#train, test, validation split values
trainS = Combine(Literal(".") + Word(numbers)).setResultsName("train_split")
testS = Combine(Literal(".") + Word(numbers)).setResultsName("test_split")
valS = Combine(Literal(".") + Word(numbers)).setResultsName("validation_split")

#Compose phrases and values together 
training = trainPhrase + trainS
testing = testPhrase + testS
val = valPhrase + valS

#Creating Optional Split phrase
ocomma = Optional(",").suppress()
split = Optional(splitKeyword + openParen + training + ocomma + testing + ocomma + val + closeParen)

#Combining READ and SPLIT keywords into one clause for combined use
read_split = read + split

#Split test
splitTest = "SPLIT (train = .8, test = .2, validation = .0)"

print(split.parseString(splitTest))

#Read with Split test
read_split_test = readTest + " "+ splitTest

print(read_split.parseString(read_split_test))

#Algorithm Definitions
algoPhrase = (Literal ("algorithm") + Literal("=")).suppress()
svmPhrase = oneOf(["svm", "SVM"])
logPhrase = oneOf(["logistic", "Logistic", "LOGISTIC"])

#Options for classifiers

#Compositions
svm = svmPhrase + Optional(openParen + closeParen)
log = logPhrase + Optional(openParen + closeParen)
algo = algoPhrase + MatchFirst([svm, log]).setResultsName("algorithm")

#define so that there can be multiple verisions of Classify
classifyKeyword = oneOf(["Classify", "CLASSIFY"]).suppress()

#Phrases to organize predictor and label column numbers
predPhrase = (Literal("predictors") + Literal("=")).suppress()
labelPhrase = (Literal("label") + Literal("=")).suppress()

#define predictor and label column numbers
predictorsDef = OneOrMore(Word(numbers) + ocomma).setResultsName("predictors")
labelDef = Word(numbers).setResultsName("label")

#combine phrases with found column numbers
preds = predPhrase + openParen + predictorsDef + closeParen
labels = labelPhrase + labelDef

classify = Optional(classifyKeyword + openParen + preds + ocomma + labels + ocomma + algo + closeParen)

classifyTest = "CLASSIFY (predictors = (1,2,3,4), label = 5, algorithm = SVM)"

print(classify.parseString(classifyTest))

#Algorithm Definitions
simplePhrase = oneOf(["simple", "SIMPLE", "Simple"])
lassoPhrase = oneOf(["lasso", "Lasso", "LASSO"])
ridgePhrase = oneOf(["ridge", "Ridge", "RIDGE"])

#Options for classifiers

#Compositions
simple = simplePhrase + Optional(openParen + closeParen)
lasso = lassoPhrase + Optional(openParen + closeParen)
ridge = ridgePhrase + Optional(openParen + closeParen)
algo = algoPhrase + MatchFirst([simple, lasso, ridge]).setResultsName("algorithm")

#define so that there can be multiple verisions of Regression
regressionKeyword = oneOf(["Regression", "REGRESSION"]).suppress()

#Phrases to organize predictor and label column numbers
predPhrase = (Literal("predictors") + Literal("=")).suppress()
labelPhrase = (Literal("label") + Literal("=")).suppress()

#define predictor and label column numbers
predictorsDef = OneOrMore(Word(numbers) + ocomma).setResultsName("predictors")
labelDef = Word(numbers).setResultsName("label")

#combine phrases with found column numbers
preds = predPhrase + openParen + predictorsDef + closeParen
labels = labelPhrase + labelDef

regression = Optional(regressionKeyword + openParen + preds + ocomma + labels + ocomma + algo + closeParen)

regressionTest = "REGRESSION (predictors = (1,2,3,4), label = 5, algorithm = simple)"

print(regression.parseString(regressionTest))

read_split_classify = read + split + classify
read_split_classify_regression = read + split + classify + regression

query1 = readTest + " " + splitTest + " " + classifyTest

print(query1)

#define a pipeline to accomplish all of the data tasks we envision
result1 = read_split_classify.parseString(query1)

#Extract relevant features from the query
filename1 = result1.filename
header1 = result1.header
sep1 = result1.sep
train1 = result1.train_split
test1 = result1.test_split
predictors1 = result1.predictors
label1 = result1.label
algo1 = str(result1.algorithm)

#Preform classification dataflow

#read file 
file1 = pd.read_csv(filename1, header = str_to_bool(header1), sep = sep1)

#predictors and labels
pred_cols = map(int, predictors1)
pred_cols = map(lambda x: x - 1, pred_cols)
label_col = int(label1) - 1

X = file1.ix[:,pred_cols]
y = file1.ix[:,label_col]

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=float(train1), test_size=float(test1))

#choose classification algorithm
if algo1.lower() == "svm":
    clf = svm.SVC()
elif algo.lower() == "logistic":
    clf = LogisticRegression()

#Train model
clf.fit(X_train, y_train)

#Performance on test data
clf.score(X_test, y_test)

