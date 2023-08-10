import graphlab
song_data = graphlab.SFrame('song_data.gl')
len(song_data)
# 1.1 million data

song_data.head()

graphlab.canvas.set_target("ipynb")

# prints how much frequently the song is played, in descending order
song_data["song"].show()

# count no. of users
users = song_data["user_id"].unique()
len(users)

training_data, test_data = song_data.random_split(0.8, seed=0)

popularity_model = graphlab.popularity_recommender.create(training_data,
                                      user_id="user_id",
                                      item_id="song")

# popularity model recommends same for everybody, based on whats popular across all people
# something like youtube trending page,which is same for all users
popularity_model.recommend(users=[users[0]])

popularity_model.recommend(users=[users[1]])

personalized_model = graphlab.item_similarity_recommender.create(training_data,
                                                                user_id="user_id",
                                                                item_id="song")

# this model suggests different songs for different users, based on their previous history.
personalized_model.recommend(users=[users[0]])

personalized_model.recommend(users=[users[1]])
# observe that both users are suggested different set of songs

# items similar to this song, will be displayed, observe that not all songs have to be usher.
personalized_model.get_similar_items(['Nice & Slow - Usher'])

if graphlab.version[:3] >= "1.6":
    model_performance = graphlab.compare(test_data, [popularity_model, personalized_model], user_sample=0.05)
    graphlab.show_comparison(model_performance,[popularity_model, personalized_model])
else:
    get_ipython().magic('matplotlib inline')
    model_performance = graphlab.recommender.util.compare_models(test_data, [popularity_model, personalized_model], user_sample=.05)

# observe that area under the curve for "popular_recommender" is way too low compared to area under curve for "similarity recommender model"



