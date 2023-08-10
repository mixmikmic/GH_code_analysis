import pandas as pd
from sklearn import datasets
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from RegressionVisualizationClasses import createRegModel, plotRegression
get_ipython().magic('matplotlib inline')
#User-adjustable parameters
rd = 3 #decimal places to round to
warn = 4 #standard deviations away that indicate a warning
alarm = 5 #standard deviations away that indicate an alarm
figSize = (20,10) #The preferred size of the output visualization
#Load the diabetes dataset and qualify it by labeling features and targets.
diabetes = datasets.load_diabetes()
df = pd.DataFrame(data=diabetes["data"][:,[2,6]],columns=["Feature 1","Feature 2"])
df["Target"] = pd.DataFrame(data=diabetes["target"],columns=["Target"])
#70% training data is usually a good size, but could be adjusted as necessary
train, test = train_test_split(df, train_size = 0.7)  
#I split the model building into several steps, mainly so that I could
#log the time to complete each step and find any bottlenecks
rModel = createRegModel("Diabetes","Dataset 1","Target",["Feature 1","Feature 2"])
timeToTT = rModel.constructXYTrainTest(train,test)
timeToModel = rModel.trainModel()
timeToPredict = rModel.makePredictions()
timeToCorrect = rModel.calcCorrectedY()
timeToBuild = rModel.buildEquation(rd)
timeToMetric = rModel.calcMetrics(rd,warn,alarm)
#Finally, a class was created to build the plots
curPlot = plotRegression(rModel, "Diabetes","Dataset 1")  
timeToPlot = curPlot.buildPlots(figSize)

