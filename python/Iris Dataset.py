from mlsql import execute

query = 'READ iris.data (separator = ",", header = False)         SPLIT (train = .8, test = .2, validation = .0)         CLASSIFY (predictors = [1,2,3,4], label = 5, algorithm = SVM)'

execute(query)

