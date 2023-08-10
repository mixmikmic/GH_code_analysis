import graphlab as gl
import numpy as np

explicit_data = gl.load_sframe("explicit_rating_data")

explicit_data = explicit_data.unstack(["user_id", "ratings"], "user/rating")

explicit_data

big_data, small_data = explicit_data.random_split(.95, seed=0)
small_data.head(10)

# This dictionary will store count of total users that liked  value for each book
normalize_dict = {}
for book in small_data:
    res = sum([1 for rate in book["user/rating"].values() if rate > 5])
    normalize_dict.setdefault(book["book_id"], 0)
    normalize_dict[book["book_id"]] = res
        

"""
Cooccurrence matrix bulit but TOO SLOW... It took almost 1 hour to compute matrix on 5% of original data 

Dictionary is built with key as book1 id and value as another dictionary conataing book2 id as key and common 
readers b/w books as value
"""
master_dict = {}
for book1 in small_data:
    # flag used to skip master_dict to add empty temp_dict(with no common users)
    flag1 = 0
    temp_dict= {}
    
    for book2 in small_data:
        if book1 == book2: continue
        # To assert, at least one user is found in common b/w book1 and book2
        flag2 = 0
            
        # Check if user rated both the movies, if yes increase the count for these two movies
        for user in book2["user/rating"].keys():
            # users that likes book1 OR book2
            book1_or_book2 = normalize_dict[book1["book_id"]] + normalize_dict[book2["book_id"]]
            
            if user in book1["user/rating"].keys():
                #if rating <= 5 skip the book(user don't like the book)
                if book1["user/rating"][user] <= 5: continue 
                        
                flag1 = 1
                if book2["book_id"] not in temp_dict:
                    temp_dict.setdefault(book2["book_id"], 0)
                temp_dict[book2["book_id"]] += 1
                flag2 = 1
                
        # Normalizing values of common users through JACCARD SIMILARITY
        if flag2 == 1: temp_dict[book2["book_id"]] /= float(book1_or_book2)
    
    if flag1 == 1:
        master_dict[book1["book_id"]] = temp_dict

small_data[0]["user/rating"].values()

np.save("cooccurrence dict.npy", master_dict)

import graphlab as gl
import numpy as np

co_dict = np.load("cooccurrence dict.npy").item()

key_list = co_dict.keys() 
value_list = co_dict.values()

arr1 = gl.SArray(key_list)
arr2 = gl.SArray(value_list)

matrix = gl.SFrame({"book1": arr1, "common": arr2})

matrix.head(3)

matrix = matrix.stack("common", new_column_name=["book2", "similarity"])

rating_data = gl.load_sframe("explicit_rating_data/")

rating_data = rating_data[rating_data["ratings"] > 5]
rating_data.materialize()

rating_data

unique_user = rating_data[rating_data["user_id"] == 276747]
unique_user.materialize()

unique_user

bought_books = list(unique_user["book_id"])

bought_books

import graphlab as gl
import numpy as np
from operator import itemgetter

rating = np.load("rating_dictionary.npy").item()
cooccur = np.load("cooccurrence dict.npy").item()

"""
Using co_dict rather than matrix SFrame (constructed using co_dict), this will make computation much more efficient 
score list store keys in the corpus and scores on the basis for reading history of user

Our cooccurrence dictionary is really sparse (5% of original data) hence we are only able to find recommendation
just for 15 users out of 100 users(for which we tried to compute recommendation).
To increase the number of users which get recommendations, cooccur dictionary must be computed for other 95% data

This function will loops over all the users present in rating dictionary and will SKIP those user for which no 
similar movies are found.

n-> denotes the maximum number of books to be recommended to a user
"""
def co_recommender(rating_dict, co_dict, userId=None, n=5):
    recom_books = {}
    
    # Rating dictionary stores user as keys and another dictionary as values
    # containing (book/corresponding ratings give by user) as key/value pair
    if userId in rating_dict.keys():
        user_rating = rating_dict[userId]
        score = []
        flag = 0
    
        # co_dict contains book_ids as keys and another dict as values containing
        # book_ids and normalized similarity between those books(as key/value pair)
        # Loop over all the books in the inventory
        for bookId,book_sim in co_dict.items():
            temp = 0
            
            # Loop over all the previouly rated book by a user and add the similarity b/w 
            # current book and EACH of the previously rated book.
            # Compute final score by dividing total number of books user has already rated
            for prev_rated in user_rating.keys(): 
                if prev_rated in book_sim.keys():
                    temp += book_sim[prev_rated]
                    
            if temp != 0:
                # To NORMALIZE score, divide score by total number of previouly rated books 
                temp /= len(user_rating)
                flag = 1
                score.append((bookId, temp))
        score = sorted(score, key=itemgetter(1), reverse=True)[0:n]
    
        if flag == 1:
            recom_books.setdefault(userId, 0)
            recom_books[userId] = score
    return recom_books[userId]
    

recom = co_recommender(rating, cooccur, userId="103541")

recom

book_data = gl.SFrame("./csv_files/BX-Books.csv")

book_data = book_data["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]
book_data.rename({"ISBN":"book_id", "Book-Title":"title", "Book-Author":"author", "Year-Of-Publication":"year",
                      "Publisher":"publisher"})

total_list_books = []
total_list_ids = []
if recom:
    for item in recom:
        bookId = item[0]
        if bookId in book_data["book_id"]:
            book_info = book_data[book_data["book_id"] == bookId][0]
            total_list_ids.append(book_info["book_id"])
            del(book_info["book_id"])
            total_list_books.append(book_info)

total_list_books



