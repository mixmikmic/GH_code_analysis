import random
import pandas
import numpy as np

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn

random.seed(42)

data = pandas.read_csv('titanic_dataset.csv')
rows = random.sample(data.index, 5)

data.ix[rows]

def preprocess(data):
    train = data.drop(['survived', 'name', 'ticket'], axis=1).values
   
    for i in range(len(train)):
        train[i][1] = 1. if train[i][1] == 'female' else 0.
  
    return train

#Prepare training data
x_train  = preprocess(data)
y_train = data['survived']

classifier = learn.LinearClassifier(n_classes=2, 
                                    feature_columns=learn.infer_real_valued_columns_from_input(x_train), 
                                    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05),
                                   model_dir='model')

classifier.fit(x_train, y_train, batch_size=128, steps=500)

#class, gender, age, sibling/spouse, parents, fare
Jack = [3,0.0, 19, 0, 0, 5.0000]
Rose = [1, 1.0, 17, 1, 2, 100.0000]
Cal = [1, 0.0, 30, 1, 0, 100.0]

test = np.array([Jack, Rose, Cal])
pred = classifier.predict(test)
prob = classifier.predict_proba(test)

answer = ['No', 'Yes']
print("Will Jack Survive? %s" % answer[pred[0]])
print("Will Rose Survive? %s" % answer[pred[1]])
print("Will Cal Survive? %s" % answer[pred[2]])

print("\nJack's Surviving Chance: %f%%" % (prob[0][1]*100))
print("Rose's Surviving Chance: %f%%" % (prob[1][1]*100))
print("Cal's Surviving Chance: %f%%" % (prob[2][1]*100))

