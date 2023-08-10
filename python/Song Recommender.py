import graphlab

song_data = graphlab.SFrame('song_data.gl/')

song_data.head()

graphlab.canvas.set_target('ipynb')

song_data['song'].show()

users = song_data['user_id'].unique()

len(users)

train_data, test_data = song_data.random_split(.8, seed = 0)

popularity_model = graphlab.popularity_recommender.create(train_data,
                                                         user_id = 'user_id',
                                                         item_id = 'song')

popularity_model.recommend(users=[users[0]])

personalized_model = graphlab.item_similarity_recommender.create(train_data,
                                                                user_id = 'user_id',
                                                                item_id = 'song')

personalized_model.recommend(users=[users[0]])

personalized_model.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club'])

get_ipython().magic('matplotlib inline')
model_performance = graphlab.recommender.util.compare_models(test_data, [popularity_model, personalized_model],
                                                            user_sample = .05)



