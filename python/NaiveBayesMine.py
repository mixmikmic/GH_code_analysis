import numpy as np

def fit(X_train,Y_train):
    result = {}
    class_values = set(Y_train)
    for curr_value in class_values:
        result[curr_value] = {}
        result["total_data"] = len(Y_train)
        curr_class_rows = (Y_train == curr_value)
        X_train_curr = X_train[curr_class_rows]
        Y_train_curr = Y_train[curr_class_rows]
        num_features = X_train.shape[1]
        result[curr_value]["total_count"] = len(Y_train_curr)
        for j in range(1,num_features+1):
            result[curr_value][j] = {}
            all_possible_values = set(X_train[:,j-1])
            for this_value in all_possible_values:
                result[curr_value][j][this_value] = (X_train_curr[:,j-1]==this_value).sum()
    return result

def probablity(dictionary,x,current_class):
    output= np.log(dictionary[current_class]["total_count"])-np.log(dictionary["total_data"])
    num_features = len(dictionary[current_class].keys())-1;
    for j in range(1,num_features+1):
        xj = x[j-1]
        count_current_class_with_value_xj = dictionary[current_class][j][xj] + 1 
        count_current_class = dictionary[current_class]["total_count"] + len(dictionary[current_class][j].keys())
        current_xj_prob = np.log(count_current_class_with_value_xj) -np.log(count_current_class)
        output = output + current_xj_prob
    return output 

def predictSinglePoint(dictionary,x):
    classes = dictionary.keys()
    best_p = -1000
    best_class = -1
    first_run = True
    for current_class in classes:
        if(current_class == "total_data"):
            continue
        p_curr_class = probablity(dictionary,x,current_class)
        if(first_run or p_curr_class > best_p):
            best_p = p_curr_class
            best_class = current_class
        first_run = False
    return best_class

def predict(dictionary,X_test):
    Y_pred = []
    for x in X_test:
        x_class = predictSinglePoint(dictionary,x)
        Y_pred.append(x_class)
    return Y_pred

def makelabelled(column):
    second_limit = column.mean()
    first_limit = 0.5 * second_limit
    third_limit = 1.5 * second_limit
    for i in range(0,len(column)):
        if(column[i]<first_limit):
            column[i] = 0
        elif(column[i] < second_limit):
            column[i] = 1
        elif(column[i]<third_limit):
            column[i] = 2
        else:
            column[i] = 3
    return column

from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target

for i in range(0,x.shape[-1]):
    x[:,i] = makelabelled(x[:,i])

from sklearn import model_selection
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(x,y,test_size=0.25,random_state=0)

dictionary = fit(X_train,Y_train)

Y_pred = predict(dictionary,X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))



