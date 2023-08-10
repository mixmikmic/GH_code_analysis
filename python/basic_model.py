import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import preprocessing
from itertools import product
import operator
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS

spark = SparkSession.builder.master('local[4]').getOrCreate()

ratings_df = pd.read_csv('/Users/ericyatskowitz/galvanize_work/MeepleFinder/data/wa_ratings_data.csv')
ratings_df = ratings_df.rename(columns={'Unnamed: 0':'Username'})
ratings_df = ratings_df.set_index('Username')
ratings_df.drop('Unnamed: 1', axis=1, inplace=True)
ind = []
for index in ratings_df.index:
    if ratings_df.loc[index, :].isnull().all() == True:
        ind.append(index)
ratings_df.drop(ind, inplace=True)
ratings_df.fillna(0, inplace=True)

board_game_index = np.load('/Users/ericyatskowitz/galvanize_work/MeepleFinder/Erics_Web_App/board_game_dict.npy').item()
board_games = dict((y,x) for x,y in board_game_index.iteritems())
user_index = np.load('/Users/ericyatskowitz/galvanize_work/MeepleFinder/Erics_Web_App/wa_user_dict.npy').item()
users = dict((y,x) for x,y in user_index.iteritems())
als_data = pd.read_csv('/Users/ericyatskowitz/galvanize_work/MeepleFinder/als_ready_wa_ratings_data.csv')
als_data.drop('Unnamed: 0', axis=1, inplace=True)

als_spark_df = spark.createDataFrame(als_data)
als_spark_df.cache()
als_model = ALS(
    itemCol='board_game',
    userCol='user',
    ratingCol='rating',
    nonnegative=True,    
    regParam=0.1,
    rank=100,
    maxIter=10
    )
als_fit_model = als_model.fit(als_spark_df)

just_ranking_info = pd.read_csv('/Users/ericyatskowitz/galvanize_work/MeepleFinder/data/just_ranking_info.csv')
just_ranking_info = just_ranking_info.set_index('Title')
predictions_array = list(product(als_data.loc[:, 'user'].unique(), just_ranking_info.index))
predictions_df = pd.DataFrame(predictions_array, columns=['user', 'board_game'])
spark_pre_predictions_df = spark.createDataFrame(predictions_df)
spark_predictions_df = als_fit_model.transform(spark_pre_predictions_df)
pred_ratings_df = spark_predictions_df.toPandas()
pred_ratings_df.fillna(0, inplace=True)
pred_ratings_df.to_csv('pred_ratings_df.csv')

bg_data_with_dummies = pd.read_csv('model_ready_bg_data.csv')
bg_data_with_dummies = bg_data_with_dummies.set_index('Title')
bg_data_with_dummies_als = bg_data_with_dummies.rename(index=board_game_index)
x = bg_data_with_dummies_als.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
normalized_als_df = pd.DataFrame(x_scaled, 
                                 index=bg_data_with_dummies_als.index, 
                                 columns=bg_data_with_dummies_als.columns)

for game in normalized_als_df.index:
    if game not in just_ranking_info.index:
        normalized_als_df.drop(game, inplace=True)
        
Y = pdist(normalized_als_df, 'cosine')
Y = squareform(Y)
bg_data_sim = pd.DataFrame(Y, index=normalized_als_df.index, columns=normalized_als_df.index)
bg_data_sim.to_csv('game_similarity_matrix.csv')



















preds_train_data = als_fit_model.transform(als_spark_df)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(preds_train_data)
print("Root-mean-square error, rank=100, maxIter=10 = " + str(rmse))

predictions_df = predictions_data.toPandas()

from scipy.spatial.distance import cosine
import operator

new_user = pd.DataFrame({'new_user': {'Wiz-War':10, 'Terra Mystica':10, 'Twilight Imperium':10}}, index=ratings_df.columns).T
new_user.fillna(0, inplace=True)

cos_sim_dict = {}
for ind in ratings_df.index:
    cos_sim_dict[ind] = cosine(ratings_df.loc[ind, :], new_user)
sorted_dict = sorted(cos_sim_dict.items(), key=operator.itemgetter(1))
top_3 = sorted_dict[:3]
top_3_keys = [user_index[top_3[i][0]] for i in xrange(len(top_3))]
user_input_df = []
for user in top_3_keys:
    user_df = pd.DataFrame(list(product([user], als_data['board_game'].unique())))
    user_df = user_df.rename(columns={0:'user', 1:'board_game'})
    user_input_df.append(spark.createDataFrame(user_df))
print top_3_keys
pred_array = np.zeros((1, len(als_data['board_game'].unique())))
for user in user_input_df:
    preds = als_fit_model.transform(user).toPandas()
    preds.sort_values('board_game', inplace=True)
    pred_array += preds['prediction'].values
top_3_games = pred_array[0].argsort()[-10:]
print top_3_games
print sorted(pred_array[0])[-10:]
games = []
for ind in top_3_games:
    games.append(board_games[ind])
new_game1 = games[0]
new_game2 = games[1]
new_game3 = games[2]
new_game4 = games[3]
new_game5 = games[4]
new_game6 = games[5]
new_game7 = games[6]
new_game8 = games[7]
new_game9 = games[8]
new_game10 = games[9]
print new_game1
print new_game2
print new_game3
print new_game4
print new_game5
print new_game6
print new_game7
print new_game8
print new_game9
print new_game10

# als_fit_model.save('als_model2')
als_model = ALSModel.load('als_model2/')

cos_sim_dict = {}
for ind in ratings_df.index:
    cos_sim_dict[ind] = cosine(ratings_df.loc[ind, :], new_user)
sorted_dict = sorted(cos_sim_dict.items(), key=operator.itemgetter(1))
top_3 = sorted_dict[:3]
top_3_keys = [user_index[top_3[i][0]] for i in xrange(len(top_3))]
user_input_df = []
for user in top_3_keys:
    user_df = pd.DataFrame(list(product([user], als_data['board_game'].unique())))
    user_df = user_df.rename(columns={0:'user', 1:'board_game'})
    user_input_df.append(spark.createDataFrame(user_df))
print top_3_keys
pred_array = np.zeros((1, len(als_data['board_game'].unique())))
for user in user_input_df:
    preds = als_model.transform(user).toPandas()
    preds.sort_values('board_game', inplace=True)
    pred_array += preds['prediction'].values
top_3_games = pred_array[0].argsort()[-10:]
print top_3_games
print sorted(pred_array[0])[-10:]
games = []
for ind in top_3_games:
    games.append(board_games[ind])
new_game1 = games[0]
new_game2 = games[1]
new_game3 = games[2]
new_game4 = games[3]
new_game5 = games[4]
new_game6 = games[5]
new_game7 = games[6]
new_game8 = games[7]
new_game9 = games[8]
new_game10 = games[9]
print new_game1
print new_game2
print new_game3
print new_game4
print new_game5
print new_game6
print new_game7
print new_game8
print new_game9
print new_game10

pred_array.shape

just_ranking_info

just_ranking_info = pd.read_csv('new_game_ratings.csv')

ratings_df[ratings_df['Gloomhaven']!=0]['Gloomhaven'].mean()

just_ranking_info

bg_ranking_data

games_in_both_dfs = []
for game in ratings_df.columns:
    if game in just_ranking_info.index:
        games_in_both_dfs.append(game)

just_ranking_info.info()

just_ranking_info.drop_duplicates(subset=['Title'], keep='first', inplace=True)

just_ranking_info['Title'] = just_ranking_info.index

just_ranking_info

len(just_ranking_info)

just_ranking_info.drop_duplicates(subset=['Title'], keep='first', inplace=True)
for game in board_game_index.keys():
    just_ranking_info['Title'].replace(to_replace=game, value=board_game_index[game], inplace=True)
geek_ratings = just_ranking_info['Geek Rating']
num_ratings = just_ranking_info['Num Ratings']

just_ranking_info.to_csv('just_ranking_info.csv')

just_ranking_info_2 = pd.read_csv('just_ranking_info.csv')

just_ranking_info_2

num_ratings = just_ranking_info['Num Ratings']
geek_ratings = just_ranking_info['Geek Rating']
avg_ratings = just_ranking_info['Avg Rating']

new_user2 = pd.DataFrame({'new_user': {'Pandemic':10, 'Agricola':10, 'Carcassonne':10}}, index=ratings_df.columns).T
new_user2.fillna(0, inplace=True)
input_games = ['Pandemic', 'Agricola', 'Carcassonne']

cos_sim_dict = {}
for ind in ratings_df.index:
    cos_sim_dict[ind] = cosine(ratings_df.loc[ind, :], new_user2)
sorted_dict = sorted(cos_sim_dict.items(), key=operator.itemgetter(1))
top_3 = sorted_dict[:3]
top_3_keys = [user_index[top_3[i][0]] for i in xrange(len(top_3))]
user_input_df = []
for user in top_3_keys:
    user_df = pd.DataFrame(list(product([user], just_ranking_info.index)))
    user_df = user_df.rename(columns={0:'user', 1:'board_game'})
    user_input_df.append(spark.createDataFrame(user_df))
count = 0
for user in user_input_df:
    preds = als_model.transform(user).toPandas()
    preds.set_index('board_game', inplace=True)
    if count == 0:
        pred_array = preds['prediction']
    else:
        pred_array += preds['prediction']
    count += 1
pred_array *= avg_ratings/10.
top_3_games = pred_array.sort_values(ascending=False)[:6][::-1].index
games = []
for ind in top_3_games:
    if board_games[ind] not in input_games:
        games.append(board_games[ind])
new_game1 = games[0]
new_game2 = games[1]
new_game3 = games[2]
print new_game1
print new_game2
print new_game3

game_rankings = pd.read_csv('new_game_ratings.csv')

game_rankings['Avg Rating'] = pd.to_numeric(game_rankings['Avg Rating'], errors='coerce')
game_rankings['Num Ratings'] = pd.to_numeric(game_rankings['Num Ratings'], errors='coerce', downcast='integer')

game_rankings.dropna(axis=0, inplace=True)

game_rankings['Num Ratings'] = game_rankings['Num Ratings'].astype(int)

game_rankings = game_rankings.set_index('Title')

game_rankings = game_rankings.rename(index=board_game_index)

game_rankings['Title'] = game_rankings.index

game_rankings['Title'] = pd.to_numeric(game_rankings['Title'], errors='coerce', downcast='integer')

game_rankings.dropna(axis=0, inplace=True)

game_rankings.drop('Title', axis=1, inplace=True)

game_rankings['Title'] = game_rankings.index

game_rankings.drop_duplicates(subset=['Title'], keep='first', inplace=True)

game_rankings.drop('Title', axis=1, inplace=True)

len(game_rankings)

len(bg_data_with_dummies_als)

game_rankings.to_csv('just_ranking_info.csv')





board_games[12707]

pred_array.sort_values(ascending=False)[:6][::-1].index

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import preprocessing
bg_data_with_dummies = pd.read_csv('model_ready_bg_data.csv')
bg_data_with_dummies = bg_data_with_dummies.set_index('Title')
bg_data_with_dummies_als = bg_data_with_dummies.rename(index=board_game_index)
x = bg_data_with_dummies_als.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
normalized_als_df = pd.DataFrame(x_scaled, 
                                 index=bg_data_with_dummies_als.index, 
                                 columns=bg_data_with_dummies_als.columns)

for game in normalized_als_df.index:
    if game not in game_rankings.index:
        normalized_als_df.drop(game, inplace=True)
        
Y = pdist(normalized_als_df, 'cosine')
Y = squareform(Y)
bg_data_sim = pd.DataFrame(Y, index=normalized_als_df.index, columns=normalized_als_df.index)
bg_data_sim.to_csv('game_similarity_matrix.csv')

x = bg_data_with_dummies_als.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
normalized_als_df = pd.DataFrame(x_scaled, index=bg_data_with_dummies_als.index, columns=bg_data_with_dummies_als.columns)

bg_data_with_dummies_als = bg_data_with_dummies.rename(index=board_game_index)

bg_data_with_dummies_als

just_ranking_info = pd.read_csv('/Users/ericyatskowitz/galvanize_work/MeepleFinder/data/just_ranking_info.csv')
just_ranking_info = just_ranking_info.set_index('Title')

just_ranking_info = just_ranking_info.set_index('Title')

board_games[7637]

for game in normalized_als_df.index:
    if game not in game_rankings.index:
        normalized_als_df.drop(game, inplace=True)

for game in game_rankings.index:
    if game not in normalized_als_df.index:
        game_rankings.drop(game, inplace=True)

len(game_rankings)

len(normalized_als_df)

game_rankings['Title'] = game_rankings.index
game_rankings.drop_duplicates(subset=['Title'], keep='first', inplace=True)
game_rankings.drop('Title', axis=1, inplace=True)

normalized_als_df['Title'] = normalized_als_df.index
normalized_als_df.drop_duplicates(subset=['Title'], keep='first', inplace=True)
normalized_als_df.drop('Title', axis=1, inplace=True)

normalized_als_df.loc[11356]

board_game_index['Monopoly']

bg_data_with_dummies_als.loc[11356]

print len(normalized_als_df)
print len(just_ranking_info)

import datetime

bg_data_with_dummies_als['Title'] = bg_data_with_dummies_als.index

bg_data_with_dummies_als['Title'] = pd.to_numeric(bg_data_with_dummies_als['Title'], errors='coerce', downcast='integer')

bg_data_with_dummies_als.dropna(axis=0, inplace=True)

bg_data_with_dummies_als.drop('Title', axis=1, inplace=True)

small_bg_data = bg_data_with_dummies.iloc[0:7000,:]

Y = pdist(normalized_als_df, 'cosine')
Y = squareform(Y)
bg_data_sim = pd.DataFrame(Y, index=normalized_als_df.index, columns=normalized_als_df.index)



for game in bg_data_sim.loc[board_game_index['Agricola'], :].sort_values()[:11].index[1:]:
    print board_games[game]

bg_data_sim = pd.read_csv('game_similarity_matrix.csv')

bg_data_sim = bg_data_sim.set_index('Title')

bg_data_sim

for game in game_rankings.index:
    if game not in bg_data_sim.index:
        print game

bg_data_sim.to_csv('game_similarity_matrix.csv')

game_rankings.to_csv('new_game_ratings.csv')

bg_data_sim.loc[12707, game_rankings.index]

Z = linkage(Y, method = 'average')

train = np.random.choice(range(len(als_data)), size=int((len(als_data)*0.8)), replace=False)

test = [ind for ind in range(len(als_data)) if ind not in train]

print len(train)
print len(test)
len(als_data) == len(train) + len(test)

als_train_data = als_data.iloc[train]
als_test_data = als_data.iloc[test]
als_train_spark_df = spark.createDataFrame(als_train_data)
als_test_spark_df = spark.createDataFrame(als_test_data)
als_spark_df.cache()
als_model = ALS(
    itemCol='board_game',
    userCol='user',
    ratingCol='rating',
    nonnegative=True,    
    regParam=0.1,
    rank=100,
    maxIter=10
    )
als_fit_model = als_model.fit(als_train_spark_df)
preds_test_data = als_fit_model.transform(als_test_spark_df)
preds = preds_test_data.toPandas()

print("Root-mean-square error, rank=100, maxIter=10 = " + str(rmse))

avg_ratings

num_ratings = game_rankings['Num Ratings']
avg_ratings = game_rankings['Avg Rating']



avg_ratings

int(preds.loc[25678, 'board_game'])

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([1, 2, 3, 4])

(arr1 + arr2)**2

product([1, 2], [2, 3], [3, 4])

indices = np.random.choice(preds.index, size=5000)

len(indices)

preds.loc[:, 'rating'].isnull().sum()

rmse_analysis = []
item_weight = [0., 1., 2., 3., 4., 5.]
cf_weight = [2., 2.5, 3.]
avg_weight = [2., 2.5, 3.]
for i_w, cf_w, avg_w in product(item_weight, cf_weight, avg_weight):
    predictions = []
    for ind in indices:
        sim_pred=0
        this_game = preds.loc[ind, 'board_game']
        user = preds.loc[ind, 'user']
        if np.isnan(preds.loc[ind, 'prediction']):
            try:
                cf_pred = (avg_ratings[int(preds.loc[ind, 'board_game'])])/cf_w
            except IndexError:
                rating_pred = avg_ratings.mean()
        else:
            cf_pred = (preds.loc[ind, 'prediction']/cf_w)
        user_df = als_train_data[als_train_data['user'] == user]
        new_user_df = user_df[user_df['rating'] >= 7]['board_game']
        count = 0
        for i, game in enumerate(new_user_df):
            if i == 0:
                try:
                    sim_pred = (1 - bg_data_sim.loc[int(game), int(this_game)])
                    count += 1
                except KeyError:
                    continue
            else:
                try:
                    sim_pred += (1 - bg_data_sim.loc[int(game), int(this_game)])
                    count += 1
                except KeyError:
                    continue
        if count != 0:
            sim_pred /= float(count)
        try:
            rating_pred = avg_ratings[int(preds.loc[ind, 'board_game'])]
        except IndexError:
            rating_pred = avg_ratings.mean()
        pred = (sim_pred*i_w) + cf_pred + (rating_pred/avg_w)
        predictions.append(pred)

    rmse = np.sqrt((((np.array(predictions)-np.array(preds.loc[:, 'rating'][indices]))**2).sum())/len(indices))
    print 'The rmse for item weight {}, cf weight {}, and avg weight {} is: {}'.format(i_w, cf_w, avg_w, rmse)
    rmse_analysis.append((i_w, cf_w, avg_w, rmse))

second_grid_search_df = pd.DataFrame(rmse_analysis, columns=['Item Weight', 'CF Weight', 'Avg Weight', 'RMSE'])

second_grid_search_df.sort_values('RMSE')













game_rankings = pd.read_csv('new_game_ratings.csv')
game_rankings = game_rankings.set_index('Title')

bg_data_sim = pd.read_csv('game_similarity_matrix.csv')
bg_data_sim = bg_data_sim.set_index('Title')

num_ratings = game_rankings['Num Ratings']
avg_ratings = game_rankings['Avg Rating']

als_train_data = als_data.iloc[0:120849]
als_test_data = als_data.iloc[120849:]

als_train_spark_df = spark.createDataFrame(als_train_data)

als_model = ALS(
    itemCol='board_game',
    userCol='user',
    ratingCol='rating',
    nonnegative=True,    
    regParam=0.1,
    rank=100,
    maxIter=10
    )
als_fit_model = als_model.fit(als_train_spark_df)

from itertools import product

from itertools import product
predictions_array = list(product(als_data.loc[:, 'user'].unique(), game_rankings.index))
predictions_df = pd.DataFrame(predictions_array, columns=['user', 'board_game'])
spark_pre_predictions_df = spark.createDataFrame(predictions_df)
spark_predictions_df = als_fit_model.transform(spark_pre_predictions_df)
pred_ratings_df = spark_predictions_df.toPandas()
pred_ratings_df.fillna(0, inplace=True)
pred_ratings_df = to_csv('pred_ratings_df.csv')

first_user_pred_array = list(product(als_data.loc[:, 'user'].unique()[:600], game_rankings.index))

first_user_pred_array_df = pd.DataFrame(first_user_pred_array, columns=['user', 'board_game'])

first_user_pred_spark_array_df = spark.createDataFrame(first_user_pred_array_df)

first_user_pred_spark_array_df.printSchema()

first_prediction_spark_df = als_fit_model.transform(first_user_pred_spark_array_df)

first_df = first_prediction_spark_df.toPandas()

second_user_pred_array = list(product(als_data.loc[:, 'user'].unique()[600:], game_rankings.index))

second_user_pred_array_df = pd.DataFrame(second_user_pred_array, columns=['user', 'board_game'])

second_user_pred_spark_array_df = spark.createDataFrame(second_user_pred_array_df)

second_user_pred_spark_array_df.printSchema()

second_prediction_spark_df = als_fit_model.transform(second_user_pred_spark_array_df)

second_df = second_prediction_spark_df.toPandas()

pred_ratings_df = spark_predictions_df.toPandas()

pred_ratings_df = first_df.append(second_df)

pred_ratings_df.fillna(0, inplace=True)
pred_ratings_df = to_csv('pred_ratings_df.csv')

pred_ratings_df = to_csv('pred_ratings_df.csv')

pred_ratings_df = pd.read_csv('pred_ratings_df.csv')
pred_ratings_df = pred_ratings_df.set_index('board_game')

pred_ratings_df[pred_ratings_df['user'] == 1]['prediction'].sort_index()

pred_ratings_df = pd.read_csv('/Users/ericyatskowitz/galvanize_work/MeepleFinder/Erics_Web_App/pred_ratings_df.csv')
pred_ratings_df = pred_ratings_df.set_index('board_game')

pred_ratings_df

pred_validation_df = pred_ratings_df[pred_ratings_df['user'] < 1100]

pred_validation_df.drop('Unnamed: 0', axis=1, inplace=True)

pred_validation_df.to_csv('pred_validation_df.csv')

import boto3
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import operator
from itertools import product

als_data = pd.read_csv('/Users/ericyatskowitz/galvanize_work/MeepleFinder/als_ready_wa_ratings_data.csv')
als_data.drop('Unnamed: 0', axis=1, inplace=True)
board_game_index = np.load('/Users/ericyatskowitz/galvanize_work/MeepleFinder/Erics_Web_App/board_game_dict.npy').item()
user_index = np.load('/Users/ericyatskowitz/galvanize_work/MeepleFinder/Erics_Web_App/wa_user_dict.npy').item()
board_games = dict((y,x) for x,y in board_game_index.iteritems())
ratings_df = pd.read_csv('/Users/ericyatskowitz/galvanize_work/MeepleFinder/Erics_Web_App/new_wa_ratings_data.csv', index_col='Username')
just_ranking_info = pd.read_csv('/Users/ericyatskowitz/galvanize_work/MeepleFinder/Erics_Web_App/new_game_ratings.csv')
just_ranking_info.set_index('Title', inplace=True)
avg_ratings = just_ranking_info['Avg Rating']
avg_ratings = avg_ratings.sort_index()
pred_validation_df = pd.read_csv('pred_validation_df.csv')
pred_validation_df = pred_validation_df.set_index('board_game')
client = boto3.client('s3')
bg_sim = client.get_object(Bucket='ericyatskowitz', Key='data/game_similarity_matrix.csv')['Body']
bg_data_sim = pd.read_csv(bg_sim)
bg_data_sim = bg_data_sim.set_index('Title')
als_test_data = als_data.iloc[120849:]

num_users =  range(1, 8)
item_weights = [0.]
cf_weights = [1.5, 2.5, 3.5]
avg_weights = [1.5, 2.5, 3.5]
for num_user, i_w, cf_w, avg_w in product(num_users, item_weights, cf_weights, avg_weights):
    predictions = []
    for pred_user in xrange(1100, 1171):
        data = als_test_data[als_test_data['user'] == pred_user]
        input_games = data.sort_values('rating')['board_game'][-3:].values
        new_user = pd.DataFrame({'new_user': 
                                 {board_games[input_games[0]]:10, 
                                  board_games[input_games[1]]:10, 
                                  board_games[input_games[2]]:10}}, 
                                index=ratings_df.columns).T
        new_user.fillna(0, inplace=True)
        cos_sim_dict = {}
        for ind in ratings_df.index[0:1100]:
            cos_sim_dict[ind] = cosine(ratings_df.loc[ind, :], new_user)
        sorted_dict = sorted(cos_sim_dict.items(), key=operator.itemgetter(1))
        top_3 = sorted_dict[:num_user]
        top_3_keys = [user_index[top_3[i][0]] for i in xrange(len(top_3))]
        user_input_df = []
        count = 0
        for user in top_3_keys:
            preds = pred_validation_df[pred_validation_df['user'] == user]['prediction'].sort_index()
            if count == 0:
                pred_array = preds
            else:
                pred_array += preds
            count += 1
        pred_array /= num_user
        count = 0
        sim_pred = 0
        for game in input_games:
            try:
                sim_pred += (1 - (bg_data_sim.loc[game, just_ranking_info.index.astype(str)].fillna(1)))
                count += 1
            except KeyError:
                continue
        if count != 0:
            sim_pred /= float(count)
            sim_pred = sim_pred.sort_index()
        else:
            sim_pred = pd.Series(0, index=sorted(just_ranking_info.index))
        new_pred_array = pd.Series(
            (pred_array.values/cf_w + 
             avg_ratings.values/avg_w + 
             sim_pred.values*i_w), 
            index=sorted(just_ranking_info.index))
        top_games = new_pred_array.sort_values(ascending=False)[:20][::-1].index
        games = []
        for ind in top_games:
            if ind not in input_games:
                games.append(ind)
        pred = []
        for new_game in games:
            try:
                pred.append(data[data['board_game'] == new_game]['rating'].values[0])
            except IndexError:
                pred.append(0)
        predictions.append(sum(pred)/float(len(games)))
    validation = np.array(predictions).mean()
    print 'The validation score for {} users, {} item-item weight, {} cf weight, and {} avg_weight is: {}'.format(num_user, i_w, cf_w, avg_w, validation)
    validation_analysis.append((num_user, i_w, cf_w, avg_w, validation))

validation_df = pd.DataFrame(validation_analysis, columns=['Num Users', 'Item-Item Weight', 'CF Weight', 'Avg Weight', 'Validation Score'])

validation_df.sort_values('Validation Score', ascending=False)

user

plt.plot(range(10), [y for x, y in rmse_analysis])
plt.xlabel("Number of users' predictions averaged")
plt.ylabel('Validation score')
plt.xticks([0, 1, 2, 3 ,4, 5, 6, 7, 8, 9])
plt.axhline(0.270350259451, linewidth=1, linestyle='dashed', color='r')

get_ipython().magic('pinfo plt.axhline')

validation_df.sort_values('Validation', ascending=False)

validation_df = pd.read_csv('/Users/ericyatskowitz/galvanize_work/MeepleFinder/data/validation_df.csv')

validation_df.rename(columns={'RMSE':'Validation'}, inplace=True)

validation_df.drop([3, 13, 47], inplace=True)

























dict1 = np.load('data/us_ratings_data_1.npy').item()
dict2 = np.load('data/us_ratings_data_2.npy').item()
dict3 = np.load('data/us_ratings_data_3.npy').item()
dict4 = np.load('data/us_ratings_data_air_1.npy').item()
dict5 = np.load('data/us_ratings_data_air_2.npy').item()
dict6 = np.load('data/us_ratings_data_old_pro_1.npy').item()
dict7 = np.load('data/us_ratings_data_old_pro_2.npy').item()

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

us_ratings_dict = merge_dicts(dict1, dict2, dict3, dict4, dict5, dict6, dict7)
us_ratings_dict_clean = dict((key, value) for key, value in us_ratings_dict.iteritems() if value != [])

len(us_ratings_dict_clean)

us_ratings = {}
for key in us_ratings_dict_clean.keys():
    count = 0
    for value in us_ratings_dict_clean[key]:
        if count == 0:
            us_ratings[key] = {value[0]: value[1]}
        else:
            us_ratings[key].update({value[0]: value[1]})
        count += 1

us_ratings_df = pd.DataFrame(us_ratings)

import pandas as pd
us_ratings_df = pd.read_csv('us_rating_df.csv')

bg_ranking_data.read_csv('just_ranking_info.csv')

geek_ratings = bg_ranking_data['Geek Rating']
num_ratings = bg_ranking_data['Num Ratings']

us_ratings_df.rename(index=user_index, columns=board_game_index, inplace=True)

us_als_data = us_ratings_df.stack().reset_index().rename(columns={'level_0':'user','level_1':'board_game', 0:'rating'})

us_als_data.to_csv('als_ready_us_ratings_df.csv', encoding='utf-8')

us_als_data.to_csv('us_als_data.csv', encoding='utf-8')

board_games = dict(enumerate(us_ratings_df.columns))
board_game_index = dict((y,x) for x,y in board_games.iteritems())
users = dict(enumerate(us_ratings_df.index))
user_index = dict((y,x) for x,y in users.iteritems())

board_game_index = np.load('board_game_dict.npy').item()
user_index = np.load('us_user_dict.npy').item()

us_als_data

for game in board_game_index.keys():
    us_als_data['board_game'].replace(to_replace=game, value=board_game_index[game], inplace=True)

for user in user_index.keys():
    us_als_data['user'].replace(to_replace=user, value=user_index[user], inplace=True)



