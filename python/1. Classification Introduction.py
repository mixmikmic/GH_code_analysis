# Preparing the input data X and the output data Y. 
# Note that the input data has 2 dimensions, since we have 2 input features.

X = [[4,1000],[2,250],[3,700],[5,600],[2,450]]
Y = [1,0,1,1,0]

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X,Y)

# Let's try classifying a new movie with R=1 and C=50!
X_new = [[1,50]]
prediction = dtree.predict(X_new)
print(prediction)

# Now let's try a movie that with R=5 and C=2000 (you know it's good :P)
X_new = [[5,2000]]
prediction = dtree.predict(X_new)
print(prediction)

