import numpy as np 
import pandas as pd
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats
plt.style.use("ggplot")

#Load the require package
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

#Load the diabetes dataset, note that all the
diabetes = pd.read_csv("diabetes.csv",sep=",")
del diabetes["Unnamed: 0"]

# Examine the covariates

#Examine the top 5 entries
diabetes.head(5)

#Short summary
diabetes.describe()

# Understand how is age distributed by plotting a histogram

# Understand how is the BMI distributed and also its mean 

# Okay lets see if men have higher or lower BMI than women

# Do that for blood pressure

# Okay now we ask, is this different significant ?
# conduct hypothesis test to see it!

# Including interaction for gender and BP and gender and BMI
diabetes["SEXxBMI"] = diabetes["SEX"]*diabetes["BMI"]
diabetes["SEXxBP"] = diabetes["SEX"]*diabetes["BP"]

#Check it works 
diabetes.head(5)

# First of all, train_test_split
from sklearn.model_selection import train_test_split

#Setup 
y = diabetes["Y"]
del diabetes["Y"]
X = diabetes

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Now we can fit our model with extra interaction variables
#Import packages
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

#Create the Model object

#fit the model using the training sets

#Make predictions using the test set

# looking into its coefficients
pd.DataFrame(regr.coef_,columns=["Coefficient"],index=diabetes.columns).T

#Print the mean squared error

#Now we can fit our model with extra interaction variables
#Import packages
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

#Create the Model object
regr_2 = linear_model.LinearRegression()

#fit the model using the training sets
regr_2.fit(X_train["AGE"].reshape(-1,1),y_train)

#Make predictions using the test set
y_pred_2 = regr_2.predict(X_test["AGE"].reshape(-1,1))

#Print the mean squared error
print("RMSE: %.2f" % np.sqrt(mean_squared_error(y_test,y_pred_2)))

#Plotting the results
plt.plot(X_test["AGE"],y_pred_2,lw=3,color="black")
plt.scatter(X_test["AGE"],y_test)

