import pandas as mypandas
from scipy import stats as mystats

myData=mypandas.read_csv('.\datasets\Complaint_Response_Time.csv')
RT=myData.Response_Time  
RT

mystats.ttest_1samp(RT,24)

RT.mean()

#p value <0.05 ==> claim is not true - Null Hypothesis H0 rejected

