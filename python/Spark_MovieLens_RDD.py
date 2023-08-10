from pyspark.mllib.recommendation import ALS
import math
import time

movie_rating = sc.textFile("ml-latest-small/ratings.csv")

header = movie_rating.take(1)[0]
rating_data = movie_rating.filter(lambda x: x!=header).map(lambda x: x.split(",")).map(lambda x: (x[0],x[1],x[2]))

# check three rows
rating_data.take(3)

train, validation, test = rating_data.randomSplit([6, 2, 2], seed = 0)

# Define a function to perform grid search and find the best ALS model
# based on the validation RMSE

def GridSearch(train, valid, num_iterations, reg_param, n_factors):
    min_rmse = float('inf')
    best_n = -1
    best_reg = 0
    best_model = None
    for n in n_factors:
        for reg in reg_param:
            model = ALS.train(train, rank = n, iterations = num_iterations, lambda_ = reg, seed = 0)
            predictions = model.predictAll(valid.map(lambda x: (x[0], x[1])))
            predictions = predictions.map(lambda x: ((x[0], x[1]), x[2]))
            rate_and_preds = valid.map(lambda x: ((int(x[0]), int(x[1])), float(x[2]))).join(predictions)
            rmse = math.sqrt(rate_and_preds.map(lambda x: (x[1][0] - x[1][1])**2).mean())
            print '{} latent factors and regularization = {}: validation RMSE is {}'.format(n, reg, rmse)
            if rmse < min_rmse:
                min_rmse = rmse
                best_n = n
                best_reg = reg
                best_model = model
                
    pred = best_model.predictAll(train.map(lambda x: (x[0], x[1])))
    pred = pred.map(lambda x: ((x[0], x[1]), x[2]))
    rate_and_preds = train.map(lambda x: ((int(x[0]), int(x[1])), float(x[2]))).join(pred)
    train_rmse = math.sqrt(rate_and_preds.map(lambda x: (x[1][0] - x[1][1])**2).mean())               
    print '\nThe best model has {} latent factors and regularization = {}:'.format(best_n, best_reg)
    print 'traning RMSE is {}; validation RMSE is {}'.format(train_rmse, min_rmse)
    return best_model

num_iterations = 10
ranks = [6, 8, 10, 12]
reg_params = [0.05, 0.1, 0.2, 0.4, 0.8]

start_time = time.time()
final_model = GridSearch(train, validation, num_iterations, reg_params, ranks)
print 'Total Runtime: {:.2f} seconds'.format(time.time() - start_time)

num_iterations = 15
ranks = [7, 8, 9]
reg_params = [0.1, 0.2, 0.3]

final_model = GridSearch(train, validation, num_iterations, reg_params, ranks)

predictions = final_model.predictAll(test.map(lambda x: (x[0], x[1]))) 
predictions = predictions.map(lambda x: ((x[0], x[1]), x[2]))
rates_and_preds = test.map(lambda x: ((int(x[0]), int(x[1])), float(x[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda x: (x[1][0] - x[1][1])**2).mean())
print 'The testing RMSE is ' + str(error)



