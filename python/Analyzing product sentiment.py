import graphlab

products = graphlab.SFrame('amazon_baby.gl/')

products.head()

products['word_count'] = graphlab.text_analytics.count_words(products['review'])

products.head()

graphlab.canvas.set_target('ipynb')

products['name'].show()

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']

print len(giraffe_reviews)
giraffe_reviews.head()

giraffe_reviews['rating'].show(view='Categorical')

products['rating'].show(view='Categorical')

#ignore all 3* reviews
products = products[products['rating'] != 3]

#positive sentiment = 4* or 5* reviews
products['sentiment'] = products['rating'] >=4

products.head()

train_data,test_data = products.random_split(.8, seed=0)

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)

sentiment_model.evaluate(test_data, metric='roc_curve')

sentiment_model.show(view='Evaluation')

giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')

giraffe_reviews.head()

giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)

giraffe_reviews.head()

giraffe_reviews[0]['review']

giraffe_reviews[1]['review']

giraffe_reviews[-1]['review']

giraffe_reviews[-2]['review']

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

for word in selected_words:
    products[word] = products['word_count'].apply(lambda word_count: 0 if word not in word_count else word_count[word])

products.head()

train_data,test_data = products.random_split(.8, seed=0)



selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=selected_words,
                                                     validation_set=test_data)

print len(selected_words_model['coefficients'])

selected_words_model['coefficients']

print selected_words_model['coefficients'].sort('value')[0]
print selected_words_model['coefficients'].sort('value')[1]
print selected_words_model['coefficients'].sort('value')[-1]

selected_words_model.evaluate(test_data, metric='roc_curve')

selected_words_model.show(view='Evaluation')

diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']

diaper_champ_reviews

diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type='probability')

diaper_champ_reviews['selected_words_model'] = selected_words_model.predict(diaper_champ_reviews, output_type='probability')

print diaper_champ_reviews.sort('predicted_sentiment')[-1]['predicted_sentiment']
print diaper_champ_reviews.sort('predicted_sentiment')[-1]

print diaper_champ_reviews.sort('selected_words_model')[-1]['selected_words_model']
print diaper_champ_reviews.sort('selected_words_model')[-1]

products['rating'].show(view='Categorical')

print max(selected_words,key=lambda word: products[word].sum())

print min(selected_words,key=lambda word: products[word].sum())



