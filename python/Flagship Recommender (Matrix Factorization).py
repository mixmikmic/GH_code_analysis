import graphlab as gl

explicit_data = gl.load_sframe("explicit_rating_data/")

#following four lines of code extract users at random who has rated books greater than 8(high rating) 
high_rated_data = explicit_data[explicit_data["ratings"] >= 8]
low_rated_data = explicit_data[explicit_data["ratings"] < 8]
train_data_1, test_data = gl.recommender.util.random_split_by_user(high_rated_data, 
                                                                         user_id="user_id", item_id="book_id")
train_data = train_data_1.append(low_rated_data)

train_data

users = test_data["user_id"].unique()

factor_exp_model = gl.factorization_recommender.create(train_data, user_id="user_id", 
                                                       item_id="book_id", target="ratings")

factor_exp_model.save("./my_models/fact_exp_model")

factor_exp_model.recommend(users=[users[0]])[0:5]

user_data = gl.load_sframe("./user_data_clean/")
book_data = gl.load_sframe("./book_data_clean/")

fact_exp_side_model = gl.factorization_recommender.create(train_data, item_id="book_id", target="ratings", 
                                                          user_data=user_data, item_data=book_data)

fact_exp_side_model.save("./my_models/fact_exp_side_model")

fact_exp_side_model.recommend(users=[users[0]])[0:5]

implicit_data = gl.load_sframe("./implicit_rating_data/")

implicit_data

fact_imp_model = gl.ranking_factorization_recommender.create(implicit_data, item_id="book_id")

fact_imp_model.save("./my_models/rank_imp_model")

fact_imp_model.recommend(users=[users[0]])

pred_imp_data = gl.load_sframe("./predicted_implicit_data/")

fact_pred_imp_model = gl.factorization_recommender.create(pred_imp_data, item_id="book_id", target="ratings")

fact_pred_imp_model.save("./my_models/fact_pred_imp_model")

fact_pred_imp_model.recommend(users=[users[0]])[0:5]

user_data = gl.load_sframe("./user_data_clean/")
book_data = gl.load_sframe("./book_data_clean/")

fact_pred_imp_side_model = gl.factorization_recommender.create(pred_imp_data, item_id="book_id", 
                                                               target="ratings", user_data=user_data, 
                                                               item_data=book_data, max_iterations=100)

fact_pred_imp_side_model.save("./my_models/fact_pred_imp_side_model")

fact_pred_imp_side_model.recommend(users=[users[0]])[0:5]

implicit_data = gl.load_sframe("./implicit_rating_data/")

combined_data = train_data.append(implicit_data)

combined_data

fact_comb_model = gl.factorization_recommender.create(combined_data, item_id="book_id", 
                                                      target="ratings", max_iterations=500)

fact_comb_model.save("./my_models/fact_comb_model")

fact_comb_model.recommend(users=[users[0]])[0:5]

fact_comb_side_model = gl.factorization_recommender.create(combined_data, item_id="book_id", target="ratings",
                                                          user_data=user_data, item_data=book_data,
                                                          max_iterations=500)

fact_comb_side_model.save("./my_models/fact_comb_side_model")

fact_comb_side_model.recommend(users=[users[0]])

import graphlab as gl

explicit_data = gl.load_sframe("./explicit_rating_data/")

fact_comb_model = gl.load_model("./my_models/fact_comb_model/")
fact_comb_side_model = gl.load_model("./my_models/fact_comb_side_model/")
fact_exp_model = gl.load_model("./my_models/fact_exp_model/")
fact_exp_side_model = gl.load_model("./my_models/fact_exp_side_model/")
fact_pred_imp_model = gl.load_model("./my_models/fact_pred_imp_model/")
fact_pred_imp_side_model = gl.load_model("./my_models/fact_pred_imp_side_model/")

# Ranking Factorization model is compared on a different basis, I must compare it by eliminating ratings column from 
# test data
rank_imp_model = gl.load_model("./my_models/rank_imp_model/")

#following four lines of code extract users at random who has rated books greater than 8(high rating) 
high_rated_data = explicit_data[explicit_data["ratings"] >= 8]
low_rated_data = explicit_data[explicit_data["ratings"] < 8]
train_data_1, test_data = gl.recommender.util.random_split_by_user(high_rated_data, 
                                                                         user_id="user_id", item_id="book_id")
train_data = train_data_1.append(low_rated_data)

gl.canvas.set_target("ipynb")

model_performance = gl.compare(test_data, [fact_pred_imp_side_model, fact_exp_side_model])
gl.show_comparison(model_performance,[fact_pred_imp_side_model, fact_exp_side_model])

model_performance = gl.compare(train_data, [fact_pred_imp_side_model, rank_imp_model], user_sample=0.05)
gl.show_comparison(model_performance,[fact_pred_imp_side_model, rank_imp_model])

fact_imp_side_model = gl.ranking_factorization_recommender.create(implicit_data, item_id="book_id", user_data=user_data,
                                                            item_data=book_data, max_iterations=200)

fact_imp_side_model.save("./my_models/rank_imp_side_model")

fact_imp_side_model.save("./my_models/rank_imp_side_model")

rank_imp_side_model = gl.load_model("./my_models/rank_imp_side_model/")

rank_imp_side_model.recommend(users=[users[0]])

rank_imp_model = gl.load_model("./my_models/rank_imp_model/")

model_performance = gl.compare(train_data, [rank_imp_side_model, rank_imp_model], user_sample=0.05)
gl.show_comparison(model_performance,[rank_imp_side_model, rank_imp_model])



