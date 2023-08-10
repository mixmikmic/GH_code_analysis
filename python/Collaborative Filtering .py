get_ipython().run_line_magic('pylab', 'inline')
import warnings
warnings.filterwarnings('ignore')

# citation: https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

def mae(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return mean_absolute_error(prediction, ground_truth)


def collaborativeFiltering(reviews_source):
    reviews = pd.read_csv(reviews_source)
    reviews['text'] = reviews['text'].str[2:-2]

    
    print("Undersampling of the dataset started--------")
    
    #Undersampling of the dataset to get a balanced dataset
    review1 = reviews[reviews['stars'] == 1][0:12000]
    review2 = reviews[reviews['stars'] == 2][0:7000]
    review3 = reviews[reviews['stars'] == 3][0:12000]
    review4 = reviews[reviews['stars'] == 4][0:12000]
    review5 = reviews[reviews['stars'] == 5][0:12000]
    frames = [review1, review2, review3,review4,review5]
    reviews = pd.concat(frames)
    
    print("Undersampling of the dataset completed--------")
    
    # converting user_id and business_id to integers for the matrix
    reviews['user_id'] = pd.factorize(reviews.user_id)[0]
    reviews['business_id'] = pd.factorize(reviews.business_id)[0]
    
    # getting the number unique users and restaurants
    unique_users = reviews.user_id.unique().shape[0]
    unique_restaurants = reviews.business_id.unique().shape[0]
    
    #splitting the dataset
    train_data, test_data = cv.train_test_split(reviews, test_size=0.20)

    #Create two user-item matrices, one for training and another for testing
    train_data_matrix = np.zeros((unique_users, unique_restaurants))
    
    print("Creation of user-item matrix started--------")
    
    # train_data_matrix
    for line in train_data.itertuples():
         train_data_matrix[line[3], line[2]] = line[5]
            
    # test_data_matrix
    test_data_matrix = np.zeros((unique_users, unique_restaurants))
    for line in test_data.itertuples():
        test_data_matrix[line[3], line[2]] = line[5]
    
    print("Creation of user-item matrix completed--------")
    
    print("Creation of similarity matrix started--------")
    
    # calculating similarity between users
    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    # calculating similarity between items
    item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
    
    print("Creation of similarity matrix completed--------")
    
    
    print("Creation of prediction matrix started--------")
    
    item_prediction = predict(train_data_matrix, item_similarity, type='item')
    user_prediction = predict(train_data_matrix, user_similarity, type='user')
    
    print("Creation of prediction matrix completed--------")
    
    print('Printing the RMSE and MAE------------' + '\n')
    
    if reviews_source == 'reviews_restaurants_text.csv':
        rating_type = 'biased rating'
    elif reviews_source == 'reviews_restaurants_text_LinearSVM.csv':
        rating_type = 'unbiased rating from Linear SVM'
    else:
        rating_type = 'unbiased rating from Naive Bayes'
    print ('Root Mean Square Error while testing the model using ' + rating_type)
    print ('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
    print ('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)) + '\n')

    print ('Root Mean Square Error while training the model using ' + rating_type)
    print ('User-based CF RMSE: ' + str(rmse(user_prediction, train_data_matrix)))
    print ('Item-based CF RMSE: ' + str(rmse(item_prediction, train_data_matrix)) + '\n')
    
    print ('Mean Absolute Error while testing the model using ' + rating_type)
    print ('User-based CF MAE: ' + str(mae(user_prediction, test_data_matrix)))
    print ('Item-based CF MAE: ' + str(mae(item_prediction, test_data_matrix)) + '\n')

    print ('Mean Absolute Error while training the model using ' + rating_type)
    print ('User-based CF MAE: ' + str(mae(user_prediction, train_data_matrix)))
    print ('Item-based CF MAE: ' + str(mae(item_prediction, train_data_matrix)) + '\n')   

collaborativeFiltering('reviews_restaurants_text.csv')

collaborativeFiltering('reviews_restaurants_text_LinearSVM.csv')

collaborativeFiltering('reviews_restaurants_text_NaiveBayes.csv')

