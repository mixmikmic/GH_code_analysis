import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import os

input_labels = [
    ["buying", ["vhigh", "high", "med", "low"]],
    ["maint", ["vhigh", "high", "med", "low"]],
    ["doors", ["2", "3", "4", "5more"]],
    ["persons", ["2", "4", "more"]],
    ["lug_boot", ["small", "med", "big"]],
    ["safety", ["low", "med", "high"]],
]

output_labels = ["unacc", "acc", "good", "vgood"]

# Load data set
data = np.genfromtxt(os.path.join('data', 'data/car.data'), delimiter=',', dtype="U")
data_inputs = data[:, :-1]
data_outputs = data[:, -1]

def str_data_to_one_hot(data, input_labels):
    """Convert each feature's string to a flattened one-hot array. """
    X_int = LabelEncoder().fit_transform(data.ravel()).reshape(*data.shape)
    X_bin = OneHotEncoder().fit_transform(X_int).toarray()
    
    attrs_names = []
    for a in input_labels:
        key = a[0]
        for b in a[1]:
            value = b
            attrs_names.append("{}_is_{}".format(key, value))

    return X_bin, attrs_names

def str_data_to_linear(data, input_labels):
    """Convert each feature's string to an integer index"""
    X_lin = np.array([[
        input_labels[a][1].index(j) for a, j in enumerate(i)
    ] for i in data])
    
    # Indexes will range from 0 to n-1
    attrs_names = [i[0] + "_index" for i in input_labels]
    
    return X_lin, attrs_names

# Take both one-hot and linear versions of input features: 
X_one_hot, attrs_names_one_hot = str_data_to_one_hot(data_inputs, input_labels)
X_linear_int, attrs_names_linear_int = str_data_to_linear(data_inputs, input_labels)

# Put that together:
X = np.concatenate([X_one_hot, X_linear_int], axis=-1)
attrs_names = attrs_names_one_hot + attrs_names_linear_int

# Outputs use indexes, this is not one-hot:
integer_y = np.array([output_labels.index(i) for i in data_outputs])

print("Data set's shape,")
print("X.shape, integer_y.shape, len(attrs_names), len(output_labels):")
print(X.shape, integer_y.shape, len(attrs_names), len(output_labels))

# Shaping the data into a single pandas dataframe for naming columns:
pdtrain = pd.DataFrame(X)
pdtrain.columns = attrs_names
dtrain = xgb.DMatrix(pdtrain, integer_y)


num_rounds = 1  # Do not use boosting for now, we want only 1 decision tree per class.
num_classes = len(output_labels)
num_trees = num_rounds * num_classes

# Let's use a max_depth of 4 for the sole goal of simplifying the visual representation produced
# (ideally, a tree would be deeper to classify perfectly on that dataset)
param = {
    'max_depth': 4,
    'objective': 'multi:softprob',
    'num_class': num_classes
}

bst = xgb.train(param, dtrain, num_boost_round=num_rounds)
print("Decision trees trained!")
print("Mean Error Rate:", bst.eval(dtrain))
print("Accuracy:", (bst.predict(dtrain).argmax(axis=-1) == integer_y).mean()*100, "%")

def plot_first_trees(bst, output_labels, trees_name):
    """
    Plot and save the first trees for multiclass classification
    before any boosting was performed.
    """
    for tree_idx in range(len(output_labels)):
        class_name = output_labels[tree_idx]
        graph_save_path = os.path.join(
            "exported_xgboost_trees", 
            "{}_{}_for_{}".format(trees_name, tree_idx, class_name)
        )

        graph = xgb.to_graphviz(bst, num_trees=tree_idx)
        graph.render(graph_save_path)

        # from IPython.display import display
        # display(graph)
        ### Inline display in the notebook would be too huge and would require much side scrolling.
        ### So we rather plot it anew with matplotlib and a fixed size for inline quick view purposes:
        fig, ax = plt.subplots(figsize=(16, 16)) 
        plot_tree(bst, num_trees=tree_idx, rankdir='LR', ax=ax)
        plt.title("Saved a high-resolution graph for the class '{}' to: {}.pdf".format(class_name, graph_save_path))
        plt.show()

# Plot our simple trees:
plot_first_trees(bst, output_labels, trees_name="simple_tree")

fig, ax = plt.subplots(figsize=(12, 7)) 
xgb.plot_importance(bst, ax=ax)
plt.show()

num_rounds = 1  # Do not use boosting for now, we want only 1 decision tree per class.
num_classes = len(output_labels)
num_trees = num_rounds * num_classes

# Let's use a max_depth of 4 for the sole goal of simplifying the visual representation produced
# (ideally, a tree would be deeper to classify perfectly on that dataset)
param = {
    'max_depth': 9,
    'objective': 'multi:softprob',
    'num_class': num_classes
}

bst = xgb.train(param, dtrain, num_boost_round=num_rounds)
print("Decision trees trained!")
print("Mean Error Rate:", bst.eval(dtrain))
print("Accuracy:", (bst.predict(dtrain).argmax(axis=-1) == integer_y).mean()*100, "%")

# Plot our complex trees:
plot_first_trees(bst, output_labels, trees_name="complex_tree")

# And their feature importance:
print("Now, our feature importance chart considers more features, but it is still not complete.")
fig, ax = plt.subplots(figsize=(12, 7)) 
xgb.plot_importance(bst, ax=ax)
plt.show()

num_rounds = 10  # 10 rounds of boosting, thus 10 trees per class. 
num_classes = len(output_labels)
num_trees = num_rounds * num_classes

param = {
    'max_depth': 20,
    'eta': 1.43,
    'objective': 'multi:softprob',
    'num_class': num_classes,
}

bst = xgb.train(param, dtrain, early_stopping_rounds=1, num_boost_round=num_rounds, evals=[(dtrain, "dtrain")])
print("Boosted decision trees trained!")
print("Mean Error Rate:", bst.eval(dtrain))
print("Accuracy:", (bst.predict(dtrain).argmax(axis=-1) == integer_y).mean()*100, "%")

# Some plot options from the doc:
# importance_type : str, default "weight"
#     How the importance is calculated: either "weight", "gain", or "cover"
#     "weight" is the number of times a feature appears in a tree
#     "gain" is the average gain of splits which use the feature
#     "cover" is the average coverage of splits which use the feature
#         where coverage is defined as the number of samples affected by the split

importance_types = ["weight", "gain", "cover"]
for i in importance_types:
    print("Importance type:", i)
    fig, ax = plt.subplots(figsize=(12, 7))
    xgb.plot_importance(bst, importance_type=i, ax=ax)
    plt.show()

