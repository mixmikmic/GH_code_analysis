import graphlab as gl
gl.canvas.set_target('ipynb')

import os
if os.path.exists('books/ratings'):
    ratings = gl.SFrame('books/ratings')
    items = gl.SFrame('books/items')
    users = gl.SFrame('books/users')
else:
    ratings = gl.SFrame.read_csv('books/book-ratings.csv')
    ratings.save('books/ratings')
    items = gl.SFrame.read_csv('books/book-data.csv')
    items.save('books/items')
    users = gl.SFrame.read_csv('books/user-data.csv')
    users.save('books/users')

ratings.show()

m = gl.recommender.create(ratings, user_id='name', item_id='book')

m

users = ratings.head(10000)['name'].unique()

recs = m.recommend(users, k=20)

sims = m.get_similar_items()

items = items.groupby('book', {k: gl.aggregate.SELECT_ONE(k) for k in ['author', 'publisher', 'year']})

num_ratings_per_book = ratings.groupby('book', gl.aggregate.COUNT)
items = items.join(num_ratings_per_book, on='book')

items.sort('Count', ascending=False)

sims = sims.join(items[['book', 'Count']], on='book')
sims = sims.sort(['Count', 'book', 'rank'], ascending=False)
sims.print_rows(1000, max_row_width=150)

implicit = ratings[ratings['rating'] >= 4]

train, test = gl.recommender.util.random_split_by_user(implicit, user_id='name', item_id='book')

train.head(5)

m = gl.ranking_factorization_recommender.create(train, 'name', 'book', target='rating', num_factors=20)

m.evaluate_precision_recall(test, cutoffs=[50])['precision_recall_overall']

new_observation_data = gl.SFrame({'name': ['Me'], 'book': ['Animal Farm'], 'rating': [5.0]})

m.recommend(users=['Me'], new_observation_data=new_observation_data)

