# import scikit-learn
from sklearn import tree

# height, weight and shoe size data
X = [
    [181,80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], 
    [166, 65, 40], [190, 90, 47], [175, 64, 39],[159, 55, 37], 
    [171,75,42], [181, 85, 43]
]
# corresponds to the data above
Y = ['male', 'male', 'female', 'female',
     'female', 'male', 'female', 'female',
     'male', 'male']

classifier = tree.DecisionTreeClassifier()

# train the decision tree on our data
classifier.fit(X,Y)

# predict the gender
prediction = classifier.predict([[190, 70, 43]])




print (prediction)



