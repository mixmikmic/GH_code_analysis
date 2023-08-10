from mlsql import execute

query = 'READ "adult.data"         SPLIT (train = .8, test = 0.2)         CLASSIFY (predictors = [1,3,11,12], label = 15, algorithm = Logistic)'

execute(query)

