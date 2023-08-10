import graphlab

all_products = graphlab.SFrame("amazon_baby.gl")

graphlab.canvas.set_target('ipynb')
all_products.show(view="Categorical")

# using "graphlab - text_analytics" lib
all_products["word count"] = graphlab.text_analytics.count_words(all_products["review"])

all_products.head()

# we choose any of (4,5) as positive review and any of (1,2) as negative. We ignore the 'rating 3'
all_products = all_products[all_products['rating'] != 3]

all_products['rating'].show(view='Categorical')
# observe rating 3 records are removed

# adding new column "sentiment" which has 1 if rating is positive and 0 if negative
all_products['sentiment'] = all_products['rating'] >= 4
all_products.head()

all_products['sentiment'].show(view="Categorical")
# which means 84% gave positive reviews

# data split
training_data,test_data = all_products.random_split(0.8,seed=0)

# uses logistic_classifier to do so
# only feature is the word_count, which affects if it has positive sentiment or negatives
sentiment_model = graphlab.logistic_classifier.create(training_data,
                                                     target='sentiment',
                                                     features=['word count'],
                                                     validation_set=test_data)

''' "confusion matrix" - is a 4 quadrant matrix, for false+,false-,true+,true-
http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/

we graph something called "roc-curve"
this is a curve between true+ and false+.
this helps us choose what is the threshold, to improve true+ and reduce false+
threshold is the value above which is classified as tumour, below which is not tumour(for ex)

so if at threshold 0.8, true+ is good,false+ is under control
but if we change threshold at 0.9, true+ is good but false+ raises quickly, we would choose treshold as 0.8
'''
sentiment_model.evaluate(test_data, metric="roc_curve")

sentiment_model.show(view="Evaluation")

#since the above "Vulli Sophie the Giraffe Teether" product is frequently reviewed, lets explore that product
giraffe_reviews = all_products[all_products['name']=="Vulli Sophie the Giraffe Teether"]

giraffe_reviews["rating"].show(view="Categorical")

all_products['rating'].show(view="Categorical")

# adding column "predicted_sentiment" which has probablity that its predicted right
giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type="probability")
giraffe_reviews.head()
# observe the third review, with rating 1.0 has predicted sentiment very low.it means our model has classified that its pretty negative review

# sort reviews based on predicted_sentiment
giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)
giraffe_reviews.head()

# since its sorted based on predicted sentiment, first reviews in table are positive and last are negative
# displaying positive
giraffe_reviews['review'][0]

giraffe_reviews['review'][1]

# showing negative review
giraffe_reviews['review'][-1]

giraffe_reviews['review'][-2]



