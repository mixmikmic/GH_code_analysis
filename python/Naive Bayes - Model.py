# All needed imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # We need to demarcate the Training and Testing set

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import matplotlib.mlab as mlab
import math # for sqrt
import scipy.stats # normal distribution
from sklearn.preprocessing import normalize
import seaborn as sn # Data visualization

# Be sure to change this to the correct path for your computer
df = pd.read_csv('/home/adityap/Actual Desktop/Ongoing/Machine Learning/Datasets/iris-species/Iris.csv')

print(df.head())
print(df.tail())

train, test = train_test_split(df, test_size = 0.2)

print(test.head())
print(train.head())

classPriors = {}

# Lets go with eqiprobable class priors - more accurate
classPriors[1] = 50/150
classPriors[2] = 50/150
classPriors[3] = 50/150

# Alternatively we could try estimating class priors from the training set
classCounts = train['Species'].value_counts()
totalCount = classCounts[0] + classCounts[1] + classCounts[2]
assert (totalCount == 120) # Cause 20% of 150 is test

#classPriors[1] = classCounts[1]/totalCount # Iris-setosa
#classPriors[2] = classCounts[2]/totalCount # Iris-virginica
#classPriors[3] = classCounts[0]/totalCount # Iris-versicolor

print("\n\nThe class priors are \n")
print( str(classPriors[1])+" "+str(classPriors[2])+" "+str(classPriors[3]) )

assert ( classPriors[1]+classPriors[2]+classPriors[3] == 1)

dfClasses = {}
means = {}
variances = {}

# Seperate training set w.r.t class labels
dfClasses["Iris-setosa"] = df.loc[df['Species'] == "Iris-setosa"]
dfClasses["Iris-virginica"] = df.loc[df['Species'] == "Iris-virginica"]
dfClasses["Iris-versicolor"] = df.loc[df['Species'] == "Iris-versicolor"]

print(dfClasses["Iris-virginica"].head()) # Try different classes

# find mu_i and var_i which are parameters for the Gaussian Event Model
means[1] = dfClasses["Iris-setosa"].mean()
variances[1] = dfClasses["Iris-setosa"].var()

means[2] = dfClasses["Iris-virginica"].mean()
variances[2] = dfClasses["Iris-virginica"].var()

means[3] = dfClasses["Iris-versicolor"].mean()
variances[3] = dfClasses["Iris-versicolor"].var()

# Feature means for Class 1, With one more indexing I can specify my desired feature
print(means[1]) 

# variances[classNo][featureNo]

# I'll now plot the gaussian curves, 
# we can see the probability distribution of each feature given a particular class.

# Length in cm
x = np.linspace(-5, 10, 1000)

fig = plt.figure()
fig.set_size_inches(10, 8)

plt.title('Feature 1\'s Likelyhood pdf vs Sepal length in cm', size='xx-large')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('PDF of likelyhood')

# Note variance is the square of the standard deviation, here sd is the required input
plt.plot(x,mlab.normpdf(x, means[1][1], math.sqrt(variances[1][1]) ), c = 'b', label = 'Iris setosa')
plt.plot(x,mlab.normpdf(x, means[2][1], math.sqrt(variances[2][1]) ), c = 'r', label = 'Iris virginica')
plt.plot(x,mlab.normpdf(x, means[3][1], math.sqrt(variances[3][1]) ), c = 'g', label = 'Iris versicolor')

plt.legend(loc=2) # 2 => leftTop ; 1 => rightTop

plt.show()

# Then Sepal Width (feature 2) for all 3 classes

fig = plt.figure()
fig.set_size_inches(10, 8)

plt.plot(x,mlab.normpdf(x, means[1][2], math.sqrt(variances[1][2]) ), c = 'b', label = 'Iris setosa')
plt.plot(x,mlab.normpdf(x, means[2][2], math.sqrt(variances[2][2]) ), c = 'r', label = 'Iris virginica')
plt.plot(x,mlab.normpdf(x, means[3][2], math.sqrt(variances[3][2]) ), c = 'g', label = 'Iris versicolor')

plt.legend(loc=2)

plt.xlabel('Sepal Width (cm)')
plt.ylabel('PDF of likelyhood')
plt.title('Feature 2\'s Likelyhood pdf vs Sepal Width in cm')
plt.show()

# Then Petal Length (feature 3) for all 3 classes

fig = plt.figure()
fig.set_size_inches(10, 8)

plt.plot(x,mlab.normpdf(x, means[1][3], math.sqrt(variances[1][3]) ), c = 'b', label = 'Iris setosa')
plt.plot(x,mlab.normpdf(x, means[2][3], math.sqrt(variances[2][3]) ), c = 'r', label = 'Iris virginica')
plt.plot(x,mlab.normpdf(x, means[3][3], math.sqrt(variances[3][3]) ), c = 'g', label = 'Iris versicolor')

plt.legend(loc=2)

plt.xlabel('Petal Length (cm)')
plt.ylabel('PDF of likelyhood')
plt.title('Feature 3\'s Likelyhood pdf vs Petal Length in cm')

plt.show()

# Then Petal Width (feature 4) for all 3 classes

fig = plt.figure()
fig.set_size_inches(10, 8)

plt.plot(x,mlab.normpdf(x, means[1][4], math.sqrt(variances[1][4]) ), c = 'b', label = 'Iris setosa')
plt.plot(x,mlab.normpdf(x, means[2][4], math.sqrt(variances[2][4]) ), c = 'r', label = 'Iris virginica')
plt.plot(x,mlab.normpdf(x, means[3][4], math.sqrt(variances[3][4]) ), c = 'g', label = 'Iris versicolor')

plt.legend(loc=2)

plt.xlabel('Petal Width (cm)')
plt.ylabel('PDF of likelyhood')
plt.title('Feature 4\'s Likelyhood pdf vs Petal Width in cm')

plt.show()

# return f(x_i | C_k) 
# i.e. the probability density of getting "x" in feature i's prob distribution given class k
def pdfFeatureGivenClass(x,i,k):
    ourCalc = (1/math.sqrt(2*math.pi*variances[k][i]))*math.exp( (-(x-means[k][i])**2) / (2*variances[k][i]))
    scipyCalc = scipy.stats.norm( means[k][i], math.sqrt(variances[k][i]) ).pdf(x)
    
    # You might see around 10^(-16)
    #print("error in our calculation : "+str(scipyCalc-ourCalc))
    return(scipyCalc)


print(pdfFeatureGivenClass(0.1,4,1)) # Try different values

# Given a string - class name, return class no
def labelofClass(x):
    return {
        'Iris-setosa': 1,
        'Iris-virginica': 2,
        'Iris-versicolor': 3,
    }.get(x, -1)    # -1 is default if x not found

labelofClass('Iris-setosa') # Just testing it

predictedProb = {}

# Posterior prob for Training set samples
for index, row in train.iterrows():
    print("\n\nFor the sample in the "+str(index)+"th row with id : "+str(row[0])+" the 4 features take the \nfollowing values in cm,\n")
    print("\nSepal Length : "+str(row[1])+
          "\nSepal Width : " +str(row[2])+
          "\nPetal Length : "+str(row[3])+
          "\nPetal Width : " +str(row[4]))
    
    assert (row[0] == index + 1) # Note id starts from 1 till 150 while index goes from 0 to 149
    
    actualClass = labelofClass(row[5])
    print("\nNote this sample is drawn from class no : "+str(actualClass))
    
    # p(C_k | \vec x) = predictedProb[indexOfX][k]
    preNormalized = {}
    
    # Iterate over 3 classes and 4 features
    for k in range(1,4):
        likelyhood = 1
        print("\n\n For class no "+str(k))
        for i in range(1,5):
            likelyhood = likelyhood * pdfFeatureGivenClass(row[i],i,k)
        print("\n \t the likelyhood is propotional to "+str(likelyhood))
        preNormalized[k] = classPriors[k]*likelyhood
        print("\n \t the posterior prob is propotional to "+str(preNormalized[k]))
        
    # To obtain actual posterior probabilities we normalize
    predictedProb[index] = normalize(np.array(list(preNormalized.values())).reshape(1,-1), norm='l1')
        
    print("\n Thus pre normalization the values were")
    print(preNormalized.values())
    
    print("\n After normalization we get the probabilities")
    print(predictedProb[index])
    

confArrayTrain = np.zeros((3,3))
# Classification using MAP rule
for index, row in train.iterrows():
    predictedClass = predictedProb[index].argmax() # 0 to 2 are the positions
    actualClass = labelofClass(row[5]) # 1 to 3 are the class labels
    confArrayTrain[actualClass-1][predictedClass] += 1
    
    print("Predicted : "+str(predictedClass)+" while true class : "+str(actualClass-1)+"\n\n")
    
print(confArrayTrain)
print("\n\nThe extent of rounding off errors are "+str(error))

fig = plt.figure()
fig.set_size_inches(5, 4)

# Plotting the confusion matrix after MAP classification on training set
dfTraincm = pd.DataFrame(confArrayTrain, [1,2,3], [1,2,3])
plt.title('Confusion Matrix on Training set')
sn.heatmap(dfTraincm, annot=True)
plt.show()
# Horizontally - Actual Classes
# Vertically - Predicted Classes

# Classifying the testing set
posteriorProb = {}
errorRounding = 0

# Calculating posterior probability
for index, row in test.iterrows():
    print("\n\nFor the sample in the "+str(index)+"th row with id : "+str(row[0])+" the 4 features take the \nfollowing values in cm,\n")
    print("\nSepal Length : "+str(row[1])+
          "\nSepal Width : " +str(row[2])+
          "\nPetal Length : "+str(row[3])+
          "\nPetal Width : " +str(row[4]))
    
    assert (row[0] == index + 1) # Note id starts from 1 till 150 while index goes from 0 to 149
    
    actualClass = labelofClass(row[5])
    print("\nNote this sample is drawn from class no : "+str(actualClass))
    
    # p(C_k | \vec x) = predictedProb[indexOfX][k]
    preNormalized = {}
    
    # Iterate over 3 classes and 4 features
    for k in range(1,4):
        likelyhood = 1
        print("\n\n For class no "+str(k))
        for i in range(1,5):
            likelyhood = likelyhood * pdfFeatureGivenClass(row[i],i,k)
        print("\n \t the likelyhood is propotional to "+str(likelyhood))
        preNormalized[k] = classPriors[k]*likelyhood
        print("\n \t the posterior prob is propotional to "+str(preNormalized[k]))

    posteriorProb[index] = normalize(np.array(list(preNormalized.values())).reshape(1,-1), norm='l1')
        
    print("\n Thus pre normalization the values were")
    print(preNormalized.values())
    
    print("\n After normalization we get the probabilities")
    print(posteriorProb[index])
    
    errorRounding += abs(1 - posteriorProb[index].sum())
    
    print("\n --------------------------------------- \n")

confArrayTest = np.zeros((3,3))


# Classification using MAP rule
for index, row in test.iterrows():
    predictedClass = posteriorProb[index].argmax() # 0 to 2 are the positions
    actualClass = labelofClass(row[5]) # 1 to 3 are the class labels
    confArrayTest[actualClass-1][predictedClass] += 1
    
    print("Predicted : "+str(predictedClass)+" while true class : "+str(actualClass-1)+"\n\n")
    
print("\n\nThe extent of rounding off errors are "+str(error))

print(confArrayTest)

fig = plt.figure()
fig.set_size_inches(5,4)

# Testing Set confusion
dfTestcm = pd.DataFrame(confArrayTest, [1,2,3], [1,2,3])
plt.title('Confusion Matrix on Testing set')
sn.heatmap(dfTestcm, annot=True)
plt.show()

print(confArrayTest)
totalTest = test.count()[0]
emperror = 0

for i in range(3):
    for j in range(3):
        if(i == j):
            continue
        emperror = emperror + confArrayTest[i][j]
        
# Unweighted empirical error
print("\n \n The Misclassification rate is "+str(emperror/totalTest)+" \n")


print(train['Species'].value_counts())
print(test['Species'].value_counts())

# take only one feature
oneFeatureSetosa = dfClasses["Iris-setosa"][['SepalLengthCm','Species']]
oneFeatureSetosa.plot.hist(bins=20)
plt.legend(loc=2)
plt.show()

plt.boxplot(oneFeatureSetosa['SepalLengthCm'].values)
plt.show()

twoFeatureSetosa = dfClasses["Iris-setosa"][['SepalLengthCm', 'SepalWidthCm', 'Species']]
x = twoFeatureSetosa['SepalLengthCm'].values
y = twoFeatureSetosa['SepalWidthCm'].values

lenEdges, widthEdges = np.linspace(2.5, 6.5, 10), np.linspace(2, 5, 10)
hist, lenEdges, widthEdges = np.histogram2d(x, y, (lenEdges, widthEdges))
xidx = np.clip(np.digitize(x, lenEdges), 0, hist.shape[0]-1)
yidx = np.clip(np.digitize(y, widthEdges), 0, hist.shape[1]-1)
c = hist[xidx, yidx]
plt.scatter(x, y, c=c)
plt.show()
plt.scatter(x, y, c='g')
plt.show()

print(c)

print(sum(train))
print("\n total no of instances in train = "+str(totalCount)+"\n\n")
x1 = sum(train)[1]/totalCount
x2 = sum(train)[2]/totalCount
x3 = sum(train)[3]/totalCount
x4 = sum(train)[4]/totalCount

# Sample mean 
xMean = np.array([x1,x2,x3,x4])

print(xMean)
print("\n")

# lets compute sample covariance matrix (unbiased)
q = np.zeros([4,4])

for i in range(totalCount):
    temp = np.array([train.iloc[i][1], train.iloc[i][2], train.iloc[i][3], train.iloc[i][4]])
    q = q + np.matmul((temp - xMean),np.transpose(temp - xMean))
    
q = q/(totalCount-1)

print(q)


# return f(\vec x) for MULTIVARIATE GUASSIAN note no given class
def pdfFeatureMultiVariate(x,k):
    return(None)

