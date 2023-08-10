from sklearn import tree
import graphviz

# Average Usage in % for the last 6h: [CPU, RAM, STORAGE]

data = [ 
 ['healthy',  45, 32, 65], 
 ['unhealthy',  87,  67, 100], 
 ['unhealthy',  100, 1, 1], 
 ['unhealthy',  76, 70, 90], 
 ['unhealthy',  1, 1, 100], 
 ['unhealthy',  31, 100, 50], 
 ['healthy',  12, 65, 39], 
 ['healthy',  20, 10, 46], 
 ['unhealthy',  100, 50, 50], 
 ['healthy',  34, 70, 37], 
 ['healthy',  1, 50, 50],
 ['unhealthy',  50, 50, 100], 
 ['healthy',  50, 1, 50],
 ['unhealthy',  1, 100, 1], 
 ['healthy',  50, 50, 1],
 ['healthy',  53, 53, 80], 
]
# state to Y
states = [row[0] for row in data]
# metrics to X
metrics = [row[1:] for row in data]
print states
print metrics

# Use a Decision Tree classifier for my tree
mytree = tree.DecisionTreeClassifier()

# train the Decision Tree with our data
mytree = mytree.fit(metrics, states)

# CHALLENGE compare their reusults and print the best one!
print("low CPU, RAM OK, LOW Storage: ") 
print(mytree.predict([[10, 80, 10]]))
print("high CPU and Storage: ") 
print(mytree.predict([[80, 10, 90]]))
print("high RAM usage: ") 
print(mytree.predict([[60, 90, 10]]))

# Visulize the decision tree
dot_data = tree.export_graphviz(mytree, 
                                feature_names=['CPU','RAM','Storage'],
                                class_names=['healthy','unhealthy'],
                                filled=True, rounded=True,
                                out_file=None) 

graphviz.Source(dot_data)

