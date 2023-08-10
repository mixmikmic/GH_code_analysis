import graphlab

products = graphlab.SFrame('amazon_baby.gl/')

products = products[products['rating'] != 3]

products['sentiment'] = products['rating'] >=4

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

selected_words_counts = [x + '_count' for x in selected_words]

products['word_count'] = graphlab.text_analytics.count_words(products['review'])

for word in selected_words:
    products[word + '_count'] = products['review'].apply(lambda r: r.count(word))

for word in selected_words:
    print word, products[word + '_count'].sum()

train_data, test_data = products.random_split(.8, seed=0)

selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=selected_words_counts,
                                                     validation_set=test_data)

selected_words_model

selected_words_model.evaluate(test_data)

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)

sentiment_model.evaluate(test_data)

diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']

diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type='probability')

diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment', ascending=False)

diaper_champ_reviews

selected_words_model.predict(diaper_champ_reviews[0:3], output_type='probability'),

for word in selected_words:
    print word, diaper_champ_reviews[2]['review'].count(word)

