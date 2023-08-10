import pandas
scores=pandas.read_csv('sat_scores.csv')
states=scores.State.values
rate=scores.Rate.values
verbal=scores.Verbal.values
math=scores.Math.values
scores.drop(51, inplace=True)
variable={'Description': ['Average Math Scores', 'Participant Rate','States', 'Average Verbal Scores'], 'Data Type': scores.dtypes, 'Mean': scores.mean()}
datadict=pandas.DataFrame(variable)
datadict.to_csv('data dictionary.csv')

import csv
scorelist=[]
with open('sat_scores.csv', 'rb') as inputdata:
    filelist=csv.reader(inputdata)
    for row in filelist:
        scorelist.append(row)

print scores

labels=scores.index
scores.reset_index(inplace=True)

states=scores.State.values
statename=list(states)

scores.dtypes

mydict_rate=pandas.Series(scores.Rate.values, index=scores.State.values).to_dict()
mydict_verbal=pandas.Series(scores.Verbal.values, index=scores.State.values).to_dict()
mydict_math=pandas.Series(scores.Math.values, index=scores.State.values).to_dict()
mydict=dict([(states, [mydict_rate[states], mydict_verbal[states], mydict_math[states]]) for states in mydict_rate])

columndict={'Rate': list(rate), 'Verbal': list(verbal), 'Math': list(math)}
columndict

print 'max rate: ', max(rate)
print 'min rate: ', min(rate)
print 'max verbal score: ', max(verbal)
print 'min verbal score: ', min(verbal)
print 'max math score: ', max(math)
print 'min math score: ', min(math)

import numpy
lst=[rate, verbal, math]
def func():
    sd=[numpy.std([y for y in x]) for x in lst]
    return sd
print func()

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

fig=plt.figure(figsize=(12,8))
axes=fig.gca()
axes.hist(rate, 10, color='blue', alpha=0.8)
axes.set_xlabel('rate', fontsize=16)
axes.set_ylabel('frequency', fontsize=16)
plt.show()

fig=plt.figure(figsize=(12,8))
axes=fig.gca()
axes.hist(math, 10, color='darkred', alpha=0.8)
axes.set_xlabel('Math Score', fontsize=16)
axes.set_ylabel('frequency', fontsize=16)
plt.show()

fig=plt.figure(figsize=(12,8))
axes=fig.gca()
axes.hist(verbal, 10, color='grey', alpha=0.8)
axes.set_xlabel('Verbal Score', fontsize=16)
axes.set_ylabel('frequency', fontsize=16)
plt.show()

fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax1.scatter(rate, verbal, c='lightblue')
ax2.scatter(rate, math, c='orange')
ax3.scatter(verbal, math, c='purple')
ax1.set_xlabel('Rate')
ax1.set_ylabel('Verbal Score')
ax2.set_xlabel('Rate')
ax2.set_ylabel('Math Score')
ax3.set_xlabel('Verbal Score')
ax3.set_ylabel('Math Score')

fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax1.boxplot(rate)
ax2.boxplot(math)
ax3.boxplot(verbal)
ax1.set_xticklabels(['Rate'])
ax2.set_xticklabels(['Math Score'])
ax3.set_xticklabels(['Verbal Score'])



