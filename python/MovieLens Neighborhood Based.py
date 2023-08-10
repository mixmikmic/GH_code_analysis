import numpy as np
import pandas as pd
import graphlab
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cross_validation import train_test_split,KFold

#prepare the data
r_cols = ['movie_id','itle','genres','user_id', 'rating', 'unix_timestamp']
ratings=pd.read_csv('ratings.csv', sep=',', encoding='latin-1')
movies=graphlab.SFrame.read_csv('movies.csv')
tags=graphlab.SFrame.read_csv('tags.csv')

#split the data into training and validation sets
train, test = train_test_split(ratings, test_size=0.2)
train=graphlab.SFrame(train)
test=graphlab.SFrame(test)


#train the Recommender Model
itemSimModel_pearson = graphlab.item_similarity_recommender.create(train, user_id='userId', item_id='movieId', target='rating', similarity_type='pearson')
itemSimModel_cosine = graphlab.item_similarity_recommender.create(train, user_id='userId', item_id='movieId', target='rating', similarity_type='cosine')
itemSimModel_pearson.evaluate_rmse(test,target='rating')
graphlab.recommender.util.compare_models(test,[itemSimModel_pearson])


#print sample from model- Top 3 for first 10 entries
itemSimModel_pearson.recommend(users=range(1,11),k=10)

pearson_eval = itemSimModel_pearson.evaluate(test)

cosine_eval = itemSimModel_cosine.evaluate(test) 

view = itemSimModel_pearson.views.overview(
        validation_set=test,
        item_data=movies)

view.show()

#Naive factorization Model
m1 = graphlab.ranking_factorization_recommender.create(train,  user_id='userId', item_id='movieId', target='rating')

m1.recommend(users=range(1,11),k=10)

#The following produces the ratings matrix for the given ranking factor recommender
#m1.predict(test)

#Model with movie information
m2 = graphlab.ranking_factorization_recommender.create(train,  user_id='userId', item_id='movieId', item_data=movies, target='rating')
m2.recommend(users=range(1,11))

#The following produces the ratings matrix for the given ranking factor recommender that adds movie genres as a side feature
#m2.predict(test)

#Model with tag information
m3 = graphlab.ranking_factorization_recommender.create(train,  user_id='userId', item_id='movieId', item_data=tags, target='rating')
m3.recommend(users=range(1,11),k=10)

#The following produces the ratings matrix for the given ranking factor recommender that adds movie tags as a side feature
#m3.predict(test)


#Model that pushes predicted ratings of unobserved user-item pairs toward 1 or below with movie genres as side feature
m4=  graphlab.ranking_factorization_recommender.create(train,  user_id='userId', item_id='movieId', item_data=movies, target='rating', unobserved_rating_value = 1)

m4.recommend(users=range(1,11),k=10)

#The following produces the ratings matrix for the given ranking factor recommender 
#m.4predict(test)


#Model that pushes predicted ratings of unobserved user-item pairs toward 1 or below with tags as side feature
m5=  graphlab.ranking_factorization_recommender.create(train,  user_id='userId', item_id='movieId', item_data=tags, target='rating', unobserved_rating_value = 1)

m5.recommend(users=range(1,11),k=10)

#The following produces the ratings matrix for the given ranking factor recommender 
#m5.predict(test)

#m1.evaluate_rmse(test,target='rating')  #'rmse_overall': 1.1997219768416447

#m2.evaluate_rmse(test,target='rating') #'rmse_overall': 1.5102909960735822

#m3.evaluate_rmse(test,target='rating') #'rmse_overall': 1.1110787691981734

#m4.evaluate_rmse(test,target='rating') #'rmse_overall': 1.467256102591114

#m5.evaluate_rmse(test,target='rating') #'rmse_overall': 1.0224394708220514

x=np.array([0,.01,.1,.25,.5,.75,.9,.99])
y=np.array([.016460668430915367,.05088695106015607,.16003760731422528,0.27976936252929974,.4133214077837332,.5950865397988541,.6098885076702547,.6931611640547327])
plt.figure(figsize=(10,8))
plt.plot(x,y,'c')

plt.title('Tuning Ranking Regularization on Explicit Target Ratings')
plt.xlabel('Ranking Regularization')
plt.ylabel('RMSE')
    
plt.show()

m_star=  graphlab.ranking_factorization_recommender.create(train,  user_id='userId', item_id='movieId', item_data=tags, target='rating', unobserved_rating_value = 1, ranking_regularization=0)
m_star.evaluate_rmse(test,target='rating') # 'rmse_overall': 1.064844763280207

view = m_star.views.overview(
        validation_set=test,
        item_data=movies)

view.show()

objects = ('PCC', 'Cosine', 'M_star')
y_pos = np.arange(len(objects))
performance = [1.1601208132503962, 3.597016790645389,1.064844763280207]
plt.figure(figsize=(10,8))
plt.bar(y_pos, performance, align='center', alpha=0.5, color='c')
plt.xticks(y_pos, objects)
plt.ylabel('RMSE')
plt.title('Model vs RMSE')
plt.show()



