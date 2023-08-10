from mlsql import execute

#Read
query1 = 'READ /home/ubuntu/notebooks/ML-SQL/dataflows/Classification/iris.data (sep="," header=False)'

execute(query1)

#Read and Split
query2 = 'READ /home/ubuntu/notebooks/ML-SQL/dataflows/Classification/iris.data (sep="," header=False)             SPLIT (train = .8, test = .2, validation = .0)'

execute(query2)

#Read Split and Classify
query3 = 'READ /home/ubuntu/notebooks/ML-SQL/dataflows/Classification/iris.data (sep="," header=False)             SPLIT (train = .8, test = .2, validation = .0)             CLASSIFY (predictors = (1,2,3,4), label = 5, algorithm = SVM)'

execute(query3)

#Read Split and Regression
query4 = 'READ /home/ubuntu/notebooks/ML-SQL/dataflows/Classification/iris.data (sep="," header=False)             SPLIT (train = .8, test = .2, validation = .0)             REGRESSION (predictors = (1,2,4), label = 3, algorithm = LASSO)'

execute(query4)

#Read Split and Classify with hyperparameters
query5 = 'READ /home/ubuntu/notebooks/ML-SQL/dataflows/Classification/iris.data (sep="," header=False)             SPLIT (train = .8, test = .2, validation = .0)             CLASSIFY (predictors = (1,2,3,4), label = 5, algorithm = SVM (gamma = 10))'

execute(query5)

