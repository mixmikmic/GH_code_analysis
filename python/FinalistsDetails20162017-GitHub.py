import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().magic('matplotlib inline')

#using the recording from March 2nd, after 2016 Finalists had ended their year of eligibility for the PMF program.
df=pd.read_csv("FinalistsDetailed1617.csv",encoding = "ISO-8859-1")

df.head(2)

a=df.loc[df2.loc[:,'Class']==2016,'Status'].value_counts()
print( "%s of %s are appointed, for a percentage of %s" %(a['Appointed'], (a['-']+a['Appointed']), a['Appointed']/(a['-']+a['Appointed'])))
print( "%s of %s are not appointed, for a percentage of %s" %(a['-'], (a['-']+a['Appointed']), a['-']/(a['-']+a['Appointed'])))
sns.countplot(x='Status',data=df.loc[df.loc[:,'Class']==2016,:]);

df2=pd.DataFrame(); #new dataframe for advanced degrees
df2['Degree']=df.loc[:,'Advanced Degree'].value_counts().index; #first column==advanced degree names
df2['Total']=df.loc[:,'Advanced Degree'].value_counts().values;
#this next section makes 2 columns for 2016 and 2017 counts of advanced degrees
ad2016=df.loc[df.loc[:,'Class']==2016,'Advanced Degree'].value_counts()
ad2017=df.loc[df.loc[:,'Class']==2017,'Advanced Degree'].value_counts()
df2['Finalists 2016']=0
df2['Finalists 2017']=0
for elm in ad2016.index:
    df2.loc[df2.loc[:,'Degree']==elm,'Finalists 2016']=ad2016[elm]
for elm in ad2017.index:
    df2.loc[df2.loc[:,'Degree']==elm,'Finalists 2017']=ad2017[elm]
#can look at the ratio of degrees from one year to the next:
df2['Ratio']=df2['Finalists 2017']/df2['Finalists 2016']
#but because the number of finalists in each year is different, this raio should be scaled:
c=df.loc[:,'Class'].value_counts()
c[2017]/c[2016]
df2['Ratio2']=df2['Ratio']-c[2017]/c[2016]
#then I can also scale the number of finalists in the 2016 record as if there had been the same number of finalists
df2['Finalists 2016 scaled']=df2['Finalists 2016']*c[2017]/c[2016]
#and look at the difference to determine which degrees have seen the most change:
df2["Finalists 2017 vs scaled 2016"]=df2['Finalists 2017']-df2['Finalists 2016 scaled']

df2.head()

fig, axes = plt.subplots(figsize=(15,3));
axes=sns.barplot(x='Degree',y='Total',
            data=df2.loc[(df2.loc[:,'Total']>=1),:].sort_values('Total'))
plt.xticks(rotation=90);
plt.ylabel('Number of Finalists')
plt.title('Advanced Degrees held by 2016-2017 PMF Finalists');
plt.savefig('PMFdegrees1617.png',bbox_inches='tight')

n=4
a=df2.Total
aT=len(a)
aTn=len(a[a<n])

print('Number of unique Advanced Degrees in the 2016-2017 Finalist list: %s'%aT)
print('Number of unique Degrees in the 2016-2017 Finalist list held by fewer than %s Finalists: %s'%(n,aTn))
print('Percent of unique Degrees held by fewer than %s Finalists: %s'%(n,round(aTn/aT*100,1)))

def expdecay(N0,L,x):
    Num=N0*np.exp(-L*x)
    return Num

fig, axes = plt.subplots(figsize=(15,3));
axes=sns.barplot(x='Degree',y='Total',
            data=df2.loc[(df2.loc[:,'Total']>=1),:].sort_values('Total'))
plt.xticks(rotation=90);
plt.ylabel('Number of Finalists')
plt.title('Advanced Degrees held by 2016-2017 PMF Finalists');
x=range(0,92)
y=[];
for X in x:
    y.append(1+expdecay(200,.2,(92-X)))
plt.plot(x,y)

n=4;
m=n;
fig, axes = plt.subplots(figsize=(8,3));
axes=sns.barplot(x='Degree',y='Finalists 2017 vs scaled 2016',
            data=df2.loc[(df2.loc[:,'Finalists 2016']>n)&(
            df2.loc[:,'Finalists 2017']>m),:].sort_values('Finalists 2017 vs scaled 2016'))
plt.xticks(rotation=90);
plt.ylabel('Number of Finalists');
plt.xlabel('Degree');
plt.title('Change in number of Finalists from 2016 to 2017\n(for advanced degrees with more than %s Finalists each year)'%(n));
plt.savefig('ADchange2016to2017_%s.png'%(n),bbox_inches='tight')

a=df.loc[(df.loc[:,'Class']==2016)&(df.loc[:,'Status']=='Appointed'),'Advanced Degree'].value_counts()
b=df.loc[(df.loc[:,'Class']==2016)&~(df.loc[:,'Status']=='Appointed'),'Advanced Degree'].value_counts()

df_success=pd.DataFrame()
df_success['Advanced Degree 2016']=df.loc[(df.loc[:,'Class']==2016),'Advanced Degree'].value_counts().index
df_success['Total']=df.loc[(df.loc[:,'Class']==2016),'Advanced Degree'].value_counts().values
df_success['Appointed']=0
df_success['Not Appointed']=0
for degree in a.index:
    df_success.loc[df_success.loc[:,'Advanced Degree 2016']==degree,'Appointed']=a[degree]
for degree in b.index:
    df_success.loc[df_success.loc[:,'Advanced Degree 2016']==degree,'Not Appointed']=b[degree]
df_success['Difference']=df_success['Appointed']-df_success['Not Appointed']
df_success['Percent']=round(df_success['Appointed']/df_success['Total']*100,1)
df_success['Percent to Average']=df_success['Percent']-57.0

df_success['Average Expectation']=df_success['Total']*57/100
df_success['Performance Relative to Expectation']=df_success['Appointed']-df_success['Average Expectation']

n=4;
m=n;
fig, axes = plt.subplots(figsize=(8,3));
#axes=sns.barplot(x='Advanced Degree 2016',y='Percent',
#            data=df_success.loc[(df_success.loc[:,'Total']>n),:].sort_values('Percent'))
axes=sns.barplot(x='Advanced Degree 2016',y='Performance Relative to Expectation',
            data=df_success.loc[(df_success.loc[:,'Total']>n),:].sort_values('Performance Relative to Expectation'))

plt.xticks(rotation=90);
plt.ylabel('Number of appointed finalists\nrelative to 57% of total finalists');
plt.xlabel('Degree');
plt.title('Number of appointed Finalists relative to expectations\nbased on 2016 appointment rate (For degrees with more than %s Finalists)'%(n));
plt.savefig('SuccessRate2016_%s.png'%(n),bbox_inches='tight')

n=5;
m=n;
fig, axes = plt.subplots(figsize=(8,3));
#axes=sns.barplot(x='Advanced Degree 2016',y='Percent',
#            data=df_success.loc[(df_success.loc[:,'Total']>n),:].sort_values('Percent'))
axes=sns.barplot(x='Advanced Degree 2016',y='Percent to Average',
            data=df_success.loc[(df_success.loc[:,'Total']>n),:].sort_values('Percent'))

plt.xticks(rotation=90);
plt.ylabel('Percent Difference\nto Average Rate (57%)');
plt.xlabel('Degree');
plt.title('Appointment success rate per degree Relative to 2016 Average Rate\n(For degrees with %s or more Finalists)'%(n));

df_success['Relative to 2017']=0.0
for degree in df_success['Advanced Degree 2016']:
    #print(degree)
    a=list(df2.loc[df2.loc[:,'Degree']==degree,'Finalists 2017 vs scaled 2016'])
    #print(a[0])
    #print(df_success.loc[df_success.loc[:,'Advanced Degree 2016']==degree,'Relative to 2017'])
    df_success.loc[df_success.loc[:,'Advanced Degree 2016']==degree,'Relative to 2017']=a
    #print(df_success.loc[df_success.loc[:,'Advanced Degree 2016']==degree,'Relative to 2017'])

len(df_success)

a=len(df_success.loc[(df_success.loc[:,'Performance Relative to Expectation']<=0)&(
        df_success.loc[:,'Relative to 2017']<=0),:])
b=len(df_success.loc[(df_success.loc[:,'Performance Relative to Expectation']>0)&(
        df_success.loc[:,'Relative to 2017']>0),:])
(a+b)/78

n=4
df_success2=df_success[df_success.loc[:,'Total']>n].sort_values('Total').copy()
df_success2.reset_index(inplace=True)
fig, ax1 = plt.subplots()
color1='blue'
color2='green'
ax1.plot(df_success2.index,df_success2['Relative to 2017'],color=color1)
ax1.set_ylabel('Change in number of Finalists from 2016 to 2017', color=color1)
ax1.tick_params('y', colors=color1)
plt.xticks(np.arange(min(df_success2.index), max(df_success2.index)+1., 1.0))
#ax1.set_xlabel(col)
ax1.set_xticklabels(df_success2['Advanced Degree 2016'],rotation=90)

ax2 = ax1.twinx()
ax2.plot(df_success2.index,df_success2['Performance Relative to Expectation'],color=color2)
ax2.set_ylabel('Performance Relative to Average Appointement Rate ', color=color2)
#ax2.yaxis.set_ticks(np.arange(0.,1.1, .1))
ax2.set_yticks(np.linspace(ax2.get_yticks()[0],ax2.get_yticks()[-1],len(ax1.get_yticks())));
ax2.tick_params('y', colors=color2)
plt.title('Comparing scaled change in number of Finalists with degree\nto appointment performance in 2016\n(for degrees with more than %s Finalists in combined years)'%(n));
plt.savefig('Performance_to_Acceptance_%s'%(n),bbox_inches='tight')

df_success



