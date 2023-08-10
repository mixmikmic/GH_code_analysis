# Import the necessary modules and libraries for this package
get_ipython().system('conda install numpy -y')
get_ipython().system('conda install scikit-learn -y')
get_ipython().system('conda install seaborn -y')
get_ipython().system('conda install matplotlib -y')

# Now import these packages
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Create a non noisy sine wave first
rng = np.random.RandomState(434)
x = np.sort(rng.uniform(0,2*np.pi,size=100 ),axis=0)
y = np.sin(x)

# and let's plot it to check
plt.scatter(x=x,y=y)
plt.show()

# now lets add some random noise
y += rng.uniform(1,-1,100)

# and plot the response
plt.scatter(x=x,y=y,edgecolor="black",
            c="green", label="Data")
plt.xlabel("Data / Predictors")
plt.ylabel("Outcome / Target")
plt.title("Our Dataset")
plt.legend()
plt.show()

get_ipython().system('conda install -c anaconda graphviz -y')
get_ipython().system('conda install -c anaconda python-graphviz -y')

# then import graphviz
import graphviz

## we will also need the tree class from sklearn
from sklearn import tree

## We are going to use this to install a package called anaconda-client using conda install
get_ipython().system('conda install anaconda-client -y')

# we use the conda command to export our environment "testNew" and save it as "cldsforall.yml"
get_ipython().system('conda env export -n testNew -f pushingTrees.yml')

# Fit regression model
X = x.reshape(-1,1)

# we'll plot one outside the foor loop so you see the decision tree
i = 1
# create the model with the desired max depth
regression_model = DecisionTreeRegressor(max_depth=i)
regression_model.fit(X, y)

# Predict using this model 
y_predict = regression_model.predict(X)

# COMMENT THIS OUT/DO NOT RUN IF YOU DO NOT HAVE GRAPHVIZ
dot_data = tree.export_graphviz(regression_model, out_file=None) 
graph = graphviz.Source(dot_data) 
graph

# now plot the fit
plt.scatter(x=x,y=y,edgecolor="black",
            c="green", label="Data")
plt.plot(x, y_predict, color="cornflowerblue",
             label=f'max_depth={i}', linewidth=2)
plt.xlabel("Data / Predictors")
plt.ylabel("Outcome / Target")
plt.title(f'max_depth={i}')
plt.show()

for i in range(2,4):

    # create the model with the desired max depth
    regression_model = DecisionTreeRegressor(max_depth=i)
    regression_model.fit(X, y)

    # Predict using this model 
    y_predict = regression_model.predict(X)

    # COMMENT THIS OUT/DO NOT RUN IF YOU DO NOT HAVE GRAPHVIZ
    # create the tree to be plotted
    dot_data = tree.export_graphviz(regression_model, out_file=None) 
    graph = graphviz.Source(dot_data) 
    # --------------------------------------------------------

    # now plot the fit
    plt.scatter(x=x,y=y,edgecolor="black",
                c="green", label="Data")
    plt.plot(x, y_predict, color="cornflowerblue",
                 label=f'max_depth={i}', linewidth=2)
    plt.xlabel("Data / Predictors")
    plt.ylabel("Outcome / Target")
    plt.title(f'max_depth={i}')
    plt.show()

## Likewise this will not work without graphviz
graph

i = 50

# create the model with the desired max depth
regression_model = DecisionTreeRegressor(max_depth=i)
regression_model.fit(X, y)

# Predict using this model 
y_predict = regression_model.predict(X)

# now plot the fit
plt.scatter(x=x,y=y,edgecolor="black",
            c="green", label="Data")
plt.plot(x, y_predict, color="cornflowerblue",
             label=f'max_depth={i}', linewidth=2)
plt.xlabel("Data / Predictors")
plt.ylabel("Outcome / Target")
plt.title(f'max_depth={i}')
plt.show()

# std_agg is 1/n of the sum of sqquares error. Don't worry about this too much,
# but it simply allows for the error in the lhs and rhs of a tree split to be calculated
# within find_better_split
import math
def std_agg(cnt, s1, s2): return ((s2/cnt) - (s1/cnt)**2)

class DecisionTree():
    
    # Initialisation function. Classes require an initialisation function,
    # which is called everytime you call a new DecisionTree() object. 
    
    # Our decision tree, within the DecisionTree() class refers to tself as self,
    # so when we are declaring self.x = x we are simply assigning the DecisionTree's
    # x values to be equal to the x argument we provide. For a decision tree
    # this will be our input data.
    def __init__(self, x, y, idxs = None, min_leaf=2):
        if idxs is None: idxs=np.arange(len(y)) # to begin with we have not assigned any input data to split sides
        self.x,self.y,self.idxs,self.min_leaf = x,y,idxs,min_leaf
        self.n,self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit() # find an initial split
        
    # Find a better split given the current splits 
    # To do this go through each tree and work out the new error in the proposed tree
    # as defined by the provided var_idx argument. 
    
    # When this function is called from find_varsplit this will go through each x point 
    # (self.c is equal to 1, i.e. the number of x points.) and find the best split for this point.
    # If the split is better than any previous splits it will update the split that occured. This 
    # occurs by working out which y_i has the greatest residual.   
    def find_better_split(self, var_idx):
        x,y = self.x.values[self.idxs,var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y,sort_x = y[sort_idx], x[sort_idx]
        rhs_cnt,rhs_sum,rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
        lhs_cnt,lhs_sum,lhs_sum2 = 0,0.,0.

        for i in range(0,self.n-self.min_leaf-1):
            xi,yi = sort_x[i],sort_y[i]
            lhs_cnt += 1; rhs_cnt -= 1
            lhs_sum += yi; rhs_sum -= yi
            lhs_sum2 += yi**2; rhs_sum2 -= yi**2
            if i<self.min_leaf or xi==sort_x[i+1]:
                continue

            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt
            if curr_score<self.score: 
                self.var_idx,self.score,self.split = var_idx,curr_score,xi
    
    # find where to split a tree into two new decision trees, i.e. where to split a branch 
    def find_varsplit(self):
        for i in range(self.c): self.find_better_split(i)
        if self.score == float('inf'): return
        x = self.split_col
        lhs = np.nonzero(x<=self.split)[0]
        rhs = np.nonzero(x>self.split)[0]
        self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs])
        self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs])

    @property
    def split_name(self): return self.x.columns[self.var_idx]
    
    @property
    def split_col(self): return self.x.values[self.idxs,self.var_idx]

    @property
    def is_leaf(self): return self.score == float('inf')
    
    # __repr__ is a specific function within classes describing what 
    # the string representation for a class should be
    def __repr__(self):
        s = f'n: {self.n}; val:{self.val}'
        if not self.is_leaf:
            s += f'; score:{self.score}; split:{self.split}; var:{self.split_name}'
        return s

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx]<=self.split else self.rhs
        return t.predict_row(xi)

# create our predictor values
x = np.arange(0,50)
x = pd.DataFrame({'x':x})

# just random uniform distributions in differnt ranges to descirbe our multivel digital signal

y1 = np.random.uniform(10,15,10)
y2 = np.random.uniform(20,25,10)
y3 = np.random.uniform(0,5,10)
y4 = np.random.uniform(30,32,10)
y5 = np.random.uniform(13,17,10)

# concatenate these to create our output data, i.e. what we are trying to predict with x
y = np.concatenate((y1,y2,y3,y4,y5))
y = y[:,None]

# quickly have a look at this data

plt.scatter(x,y)
plt.plot(x,y, 'o')
plt.title("Scatter plot of x vs. y")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

xi = x # initialization of input
yi = y # initialization of target
# x,y --> use where no need to change original y
ei = 0 # initialization of error
n = len(yi)  # number of rows
predf = 0 # initial prediction 0

for i in range(20): # like n_estimators
    
    # find our initial tree. On iteration 1 this will create the first split to the dataset.
    # After the fist iteration, it will use the new yi which is the error from the previous tree
    # and again trigger finding the best split within this by calling find_varsplit first as 
    # this tree is created, which in turn triggers find_better_split
    tree = DecisionTree(xi,yi)
    
    # where was that split
    r = np.where(xi == tree.split)[0][0]    
    
    left_idx = np.where(xi <= tree.split)[0]
    right_idx = np.where(xi > tree.split)[0]
    
    # what is the mean either side of the split, as this will then 
    # be how much we correct our prediciton by
    predi = np.zeros(n)
    np.put(predi, left_idx, np.repeat(np.mean(yi[left_idx]), r))  # replace left side mean y
    np.put(predi, right_idx, np.repeat(np.mean(yi[right_idx]), n-r))  # right side mean y
    
    predi = predi[:,None]  # make long vector (nx1) in compatible with y
    predf = predf + predi  # final prediction is previous prediction value + new prediction of residual
    
    ei = y - predf  # needed originl y here as residual always from original y    
    yi = ei # update yi as residual to reloop
    
    
    # plotting after prediction
    xa = np.array(x.x) # column name of x is x 
    order = np.argsort(xa)
    xs = np.array(xa)[order]
    ys = np.array(predf)[order]
    
    #epreds = np.array(epred[:,None])[order]

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = (13,2.5))

    ax1.plot(x,y, 'o')
    ax1.plot(xs, ys, 'r')
    ax1.set_title(f'Prediction (Iteration {i+1})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y / y_pred')

    ax2.plot(x, ei, 'go')
    ax2.set_title(f'Residuals vs. x (Iteration {i+1})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Residuals')

plt.show()

