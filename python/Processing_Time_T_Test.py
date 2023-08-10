import pandas as mypanda
from scipy import stats as mystats

myData=mypanda.read_csv("'.\datasets\PO_Processing.csv")
#myData

PT=myData.Processing_Time
PT

mystats.ttest_1samp(PT,40)

PT.mean()

#Conclusion is H0 rejected since p<0.05 ==> not satisfied



