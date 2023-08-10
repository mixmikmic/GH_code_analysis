import textblob
from textblob import TextBlob

sen = TextBlob("this is an amazing to use.had a great fun")
sen.sentences
sen.correct()
sen.sentiment

train = [
     ('I love this sandwich.', 'pos'),
     ('this is an amazing place!', 'pos'),
     ('I feel very good about these beers.', 'pos'),
     ('this is my best work.', 'pos'),
     ("what an awesome view", 'pos'),
     ('I do not like this restaurant', 'neg'),
     ('I am tired of this stuff.', 'neg'),
     ("I can't deal with this", 'neg'),
     ('he is my sworn enemy!', 'neg'),
     ('my boss is horrible.', 'neg')
 ]
test = [
     ('the beer was good.', 'pos'),
     ('I do not enjoy my job', 'neg'),
     ("I ain't feeling dandy today.", 'neg'),
     ("I feel amazing!", 'pos'),
     ('Gary is a friend of mine.', 'pos'),
     ("I can't believe I'm doing this.", 'neg')
 ]

from textblob.classifiers import NaiveBayesClassifier

c1 = NaiveBayesClassifier(train)

#c1.classify("This is an amazing library")
a = "This is an amazing library"
c1.classify(a)

prob_list = c1.prob_classify("This is horrible")
prob_list.max()    #gives maximum output (final result)

round(prob_list.prob("neg"),3)  #it's useful for find how much positive && how much negative

c1.accuracy(train) #test and train 

c1.accuracy(test)



