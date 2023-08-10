import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cleveland = pd.read_csv("CLE-season.csv",index_col=0,parse_dates=[1])

cleveland.head()

cleveland.info()

cleveland['Diff'] = cleveland.apply(lambda row : row['Tm'] - row['Opp'],axis=1)

cleveland.head()

cleveland['Diff'].plot(kind='bar',figsize=(15,5))

cleveland.loc[39]

def fun(x):
    return x**2

fun(3)

# Define a function in one line using the lambda keyword and name the function fun
fun = lambda x : x**2

fun(3)

golden_state = pd.read_csv("GS-season.csv",index_col=0,parse_dates=[1])

golden_state.head()

golden_state['Diff'] = golden_state.apply(lambda row : row['Tm'] - row['Opp'],axis=1)

golden_state['Diff'].plot(kind='bar',figsize=(15,5))

golden_state.loc[4]

plt.subplot(2,1,1)
cleveland['Diff'].plot(kind='bar',figsize=(15,5),title='Cleveland')
plt.subplot(2,1,2)
golden_state['Diff'].plot(kind='bar',figsize=(15,5),title='Golden State')
plt.tight_layout()

cleveland['W/L Diff'] = cleveland.apply(lambda row : row['W'] - row['L'],axis=1)
golden_state['W/L Diff'] = golden_state.apply(lambda row : row['W'] - row['L'],axis=1)

get_ipython().magic('pinfo pd.concat')

win_diff = pd.concat([cleveland['W/L Diff'],golden_state['W/L Diff']],axis=1)

win_diff.columns = ['Cleveland','Golden State']

win_diff.head()

win_diff.info()

win_diff.plot()

