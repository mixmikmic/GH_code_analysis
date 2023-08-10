import graphlab as gl

user_data = gl.load_sframe("./user_data_clean/")

# remove rows where country is not mentioned in location
fil = []
for item in user_data["location"]:
    temp = item.split(",")
    if len(temp) <= 2 or temp[2] == "":
        fil.append(False)
    else:
        fil.append(True)
fil = gl.SArray(data=fil)

user_data = user_data[fil]

# loacations where city is not mentioned replace states with name of their country rather than excluding them
# and convert a complete string of location to a list of strings containg city name and country name as elements
def modify(st):
    st = st.split(",")
    if st[1] == " " or st[1] == " n/a":
        st[1] = st[2]
    del(st[0])
    st_0 = st[0].strip() 
    st_1 = st[1].strip()
    lis = [];
    lis.append(st_0)
    lis.append(st_1)
    return lis

user_data["location"] = user_data["location"].apply(modify)

user_data.head()

book_data = gl.load_sframe("./book_data_clean/")

book_data.head()

len(book_data), len(user_data)

rating_data = gl.load_sframe("./explicit_rating_data/")

len(rating_data)

rating_data.head()

#select only those rows in rating_data whose user-ids matches with user data ids
rating_data = rating_data.filter_by(user_data["user_id"], "User-ID")

len(rating_data)

#select only those rows in user data for which user ids are present in rating data
user_data = user_data.filter_by(rating_data["User-ID"], "user_id")

len(user_data)

#Do the same with book data
rating_data = rating_data.filter_by(book_data["book_id"], "ISBN")

len(rating_data)

book_data = book_data.filter_by(rating_data["ISBN"], "book_id")

len(book_data)

rating_data.rename({"User-ID":"user_id", "ISBN":"book_id", "Book-Rating":"ratings"})

#join all three datasets on common user_ids and book_ids
complete_data = rating_data.join(user_data, on="user_id")

complete_data = complete_data.join(book_data, on="book_id")

#list type columns not accepted by linear regression model, therfore need to convert locations back to string
def modify(lis):
    st = ""
    flag = 0
    for i in lis:
        if flag == 0:
            st += i
            st += ", "
            flag = 1
        else:
            st += i
    return st

complete_data["location"] = complete_data["location"].apply(modify)

complete_data

#data on which model is to be trained contains 55862 rows. To ensure that model will perform well only five columns
#are choosen to be used as feature values. If there would have been more data "title" column could also be included
#in feature columns
features_cols = ["publisher", "age", "location", "title", "author", "year"]

#following four lines of code extract users at random who has rated books greater than 8(high rating) 
high_rated_data = complete_data[complete_data["ratings"] >= 8]
low_rated_data = complete_data[complete_data["ratings"] < 8]
train_data_1, test_data = gl.recommender.util.random_split_by_user(high_rated_data, 
                                                                         user_id="user_id", item_id="book_id")
train_data = train_data_1.append(low_rated_data)

#prototype model trained over 80% of availble data(train data) for evaluation (20% test data)
#prototype_model = gl.linear_regression.create(train_data, features=features_cols, 
#                                              target="ratings", max_iterations=3000, verbose=True)

prototype_model.evaluate(test_data)

import math
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.plot(test_data["age"], test_data["ratings"], ".",
        test_data["age"], prototype_model.predict(test_data), ".")

len(prototype_model.coefficients)

#regression_model = gl.linear_regression.create(complete_data, features=features_cols, target="ratings", 
#                                               max_iterations=5000)

regression_model.evaluate(test_data)

regression_model.save("./regression_model_file")

import graphlab as gl

regression_model = gl.load_model("./regression_model_file/")

book_data = gl.load_sframe("./book_data_clean/")

implicit_data = gl.load_sframe("./implicit_rating_data/")

book_data.filter_by(implicit_data["ISBN"], "book_id")

from operator import itemgetter
"""
This function takes as argument user's age and location(consisting state) and outputs two lists one containing ids
of recommended books and other list contains title of recommended books. Currently, it only choose a movie among 
first 3000 movies from IMPLICIT test dataset having total movies 45000.(Note that model was trained on explicit 
dataset which is different from implicit dataset).
Count of movies can be increased(by modifiying max variable) if required to search among more movies, but it will take considerable time
depending on the machine this function is evaluated upon.
"""
def predict(location, age, search_over=3000):
    predicted_ratings = []
    count = 0
    for book in book_data:
        if count == search_over:
            break
        count += 1
        book["location"] = location
        book["age"] = age
        rating = regression_model.predict(book)[0]
        if rating >= 8.0:
            predicted_ratings.append((book["book_id"], rating))
    
    predicted_ratings = sorted(predicted_ratings, key=itemgetter(1), reverse=True)

    #recommeded books in decresing values of ratings
    recommended_books_id = []
    for i in range(5):
        recommended_books_id.append(predicted_ratings[i][0])

    recommended_books = []
    for book in recommended_books_id:
        for item in book_data:
            if book in item["book_id"]:
                del(item["book_id"])
                recommended_books.append(item)
                break
    return recommended_books_id, recommended_books

ids, books = predict("delhi, india", 21)

books



