#######################################################
# Script:
#    testPerf.py
# Usage:
#    python testPerf.py <input_file> <output_file>
# Description:
#    Get the prediction based on training data model
#    Pass 1: prediction based on hours in a week
# Authors:
#    Jasmin Nakic, jnakic@salesforce.com
#    Samir Pilipovic, spilipovic@salesforce.com
#######################################################

import sys
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib

# Imports required for visualization (plotly)
import plotly.graph_objs as go
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Script debugging flag
debugFlag = False

# Feature lists for different models
simpleCols = ["dateFrac"]
trigCols = ["dateFrac", "weekdaySin", "weekdayCos", "hourSin", "hourCos"]
hourDayCols  = ["dateFrac", "isMonday", "isTuesday", "isWednesday", "isThursday", "isFriday", "isSaturday", "isSunday",
                "isHour0", "isHour1", "isHour2", "isHour3", "isHour4", "isHour5", "isHour6", "isHour7",
                "isHour8", "isHour9", "isHour10", "isHour11", "isHour12", "isHour13", "isHour14", "isHour15",
                "isHour16", "isHour17", "isHour18", "isHour19", "isHour20", "isHour21", "isHour22", "isHour23"]
hourWeekCols = ["dateFrac"]
for d in range(0,7):
    for h in range(0,24):
        hourWeekCols.append("H_" + str(d) + "_" + str(h))

# Add columns to the existing array and populate with data
def addColumns(dest, src, colNames):
    # Initialize temporary array
    tmpArr = np.empty(src.shape[0])
    cols = 0
    # Copy column content
    for name in colNames:
        if cols == 0: # first column
            tmpArr = np.copy(src[name])
            tmpArr = np.reshape(tmpArr,(-1,1))
        else:
            tmpCol = np.copy(src[name])
            tmpCol = np.reshape(tmpCol,(-1,1))
            tmpArr = np.append(tmpArr,tmpCol,1)
        cols = cols + 1
    return np.append(dest,tmpArr,1)
#end addColumns

# Get prediction using saved linear regression model
def getPredictions(data,colList,modelName):
    # Initialize array
    X = np.zeros(data.shape[0])
    X = np.reshape(X,(-1,1))

    # Add columns
    X = addColumns(X,data,colList)

    if debugFlag:
        print("X 0: ", X[0:5])

    Y = np.copy(data["cnt"])
    if debugFlag:
        print("Y 0: ", Y[0:5])

    model = joblib.load(modelName)
    P = model.predict(X)
    print("SCORE values: ", model.score(X,Y))
    if debugFlag:
        print("P 0-5: ", P[0:5])

    return P
#end getPredictions

# Write predictions to the output file
def writeResult(output,data,p1,p2,p3,p4):
    # generate result file
    result = np.array(
        np.empty(data.shape[0]),
        dtype=[
            ("timeStamp","|S19"),
            ("dateFrac",float),
            ("isHoliday",int),
            ("isSunday",int),
            ("cnt",int),
            ("predSimple",int),
            ("predTrig",int),
            ("predHourDay",int),
            ("predHourWeek",int)
        ]
    )

    result["timeStamp"]    = data["timeStamp"]
    result["dateFrac"]     = data["dateFrac"]
    result["isHoliday"]    = data["isHoliday"]
    result["isSunday"]     = data["isSunday"]
    result["cnt"]          = data["cnt"]
    result["predSimple"]   = p1
    result["predTrig"]     = p2
    result["predHourDay"]  = p3
    result["predHourWeek"] = p4

    if debugFlag:
        print("R 0-5: ", result[0:5])
    hdr = "timeStamp\tdateFrac\tisHoliday\tisSunday\tcnt\tpredSimple\tpredTrig\tpredHourDay\tpredHourWeek"
    np.savetxt(output,result,fmt="%s",delimiter="\t",header=hdr,comments="")
#end writeResult

# Start
inputFileName = "test_data.txt"
outputFileName = "test_hourly.txt"

# All input columns - data types are strings, float and int
inputData = np.genfromtxt(
    inputFileName,
    delimiter='\t',
    names=True,
    dtype=("|U19","|U10",int,float,int,float,float,int,float,float,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int,
           int,int,int,int,int,int,int,int,int,int
    )
)

P1 = getPredictions(inputData,simpleCols,"modelSimple")
P2 = getPredictions(inputData,trigCols,"modelTrig")
P3 = getPredictions(inputData,hourDayCols,"modelHourDay")
P4 = getPredictions(inputData,hourWeekCols,"modelHourWeek")

writeResult(outputFileName,inputData,P1,P2,P3,P4)

# Load the test data from file generated above using correct data types
results = np.genfromtxt(
    outputFileName,
    dtype=("|U19",float,int,int,int,int,int,int,int),
    delimiter='\t',
    names=True
)

# Examine results
print("Shape:", results.shape)
print("Columns:", len(results.dtype.names))
print(results[1:5])

# Generate chart with predicitons based on test data (using plotly)
print("Plotly version", __version__) # requires plotly version >= 1.9.0
init_notebook_mode(connected=True)

set1 = go.Bar(
    x=results["dateFrac"],
    y=results["cnt"],
#    marker=dict(color='blue'),
    name='Actual'
)
set2 = go.Bar(
    x=results["dateFrac"],
    y=results["predHourWeek"],
#    marker=dict(color='crimson'),
    opacity=0.6,
    name='Prediction'
)
barData = [set1, set2]
barLayout = go.Layout(barmode='group', title="Prediction vs. Actual")

fig = go.Figure(data=barData, layout=barLayout)
iplot(fig)



