import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
#'s' represents stored data, 'h' represents hidden data
sig = pd.DataFrame(data={'Time':range(17),
                         'Value':[1,1,1,1,1,2,2,2,3,2,3,3,2,2,2,2,2],
                         'Flag':['s','h','h','h','s','s','h','s','s',
                                 's','s','s','s','h','h','h','s']})
plt.figure(0)
#Plot the stored data as blue dots
ax = sig[sig['Flag'] == 's'].plot(x='Time',y='Value',c='b',
                                  kind='scatter',label='Stored')
#Plot the hidden data as red Xs
sig[sig['Flag'] == 'h'].plot(x='Time',y='Value',c='r',marker='x',
                             kind='scatter',label='Hidden',ax=ax)
#Adjust the axis and legend
ax.legend(loc='upper left')
plt.axis([-1,18,0.5,3.5]);

sig2 = pd.DataFrame(data={'Time':range(17),
                          'VarA':[1,None,None,None,1,2,None,2,3,2,3,3,2,None,None,None,2],
                          'VarB':[4,5,None,None,None,None,5,6,None,None,None,6,4,5,6,None,6]})
plt.figure(1)
#Plot 'raw' Variable A using blue dots
ax = sig2.plot(x='Time',y='VarA',c='b',kind='scatter',label='Variable A')
#Plot 'raw' Variable B using red dots
sig2.plot(x='Time',y='VarB',c='r',kind='scatter',label='Variable B',ax=ax)
#Adjust the axis and legend
ax.legend(loc='upper left')
plt.axis([-1,18,0.5,10]);                       

