get_ipython().magic('matplotlib inline')

from IPython.core.pylabtools import figsize
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Loading matlab styles from
# https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
import json
s = json.load(open("bmh_matplotlibrc.json"))
mpl.rcParams.update(s)

get_ipython().system('cat bmh_matplotlibrc.json')

data = pd.read_csv('data/titanic/train.csv')

# Add column to determine child man and females
def is_woman_child_or_man(passenger):
    age, sex = passenger
    if age < 16:
        return "child"
    else:
        return 'man' if sex == 'male' else 'female'
        
data["Sex"].unique()
data["Who"] = data[["Age", "Sex"]].apply(is_woman_child_or_man, axis=1)
data["Embarked"] = data['Embarked'].map({"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"})
data = data.drop(['Ticket','Cabin'], axis=1) 
data.dropna()
data.head()

## Individual plots
data['Sex'].value_counts().plot(kind='bar', alpha=.3)

# draw vertical line from (70,100) to (70, 250)
plt.plot([70, 70], [100, 250], 'k-', lw=2)

# draw diagonal line from (70, 90) to (90, 200)
plt.plot([70, 90], [90, 200], 'k-')

data['Age'][data['Sex'] == 'male'].plot(kind='kde')    
# data['Age'][data['Sex'] == 'female'].plot(kind='kde')
data['Age'][data['Sex'] == 'female'].plot(kind='kde', ls='dashed')

data['Pclass'].value_counts().plot(kind='bar', alpha=.3)

data['Age'][data['Pclass'] == 1].plot(kind='kde')    
data['Age'][data['Pclass'] == 2].plot(kind='kde')
data['Age'][data['Pclass'] == 3].plot(kind='kde')

# TODO: x labels for bar charts
# TODO: Fill area under 
fig = plt.figure(figsize=(18,6)) 

plot_1 = plt.subplot2grid((2,3),(1,0))              
data['Sex'].value_counts().plot(kind='bar', alpha=.3)
plt.title("Distribution of Gender")  
locs, labels = plt.xticks()
plt.setp(labels, rotation=0)

plot_2 = plt.subplot2grid((2,3),(0,1), colspan=2)
data['Age'][data['Sex'] == 'male'].plot(kind='kde')    
data['Age'][data['Sex'] == 'female'].plot(kind='kde')
plt.xlabel("Age")    
plt.title("Age Distribution by gender")
plt.legend(('Male', 'Female'),loc='best') 

plot_3 = plt.subplot2grid((2,3),(0,0))              
data['Pclass'].value_counts().plot(kind='bar', alpha=.3)
plt.title("Distribution of Classes")

plot_4 = plt.subplot2grid((2,3),(1,1), colspan=2)
a = data['Age'][data['Pclass'] == 1]
# data['Age'][data['Pclass'] == 1].plot(kind='kde')    
data['Age'][data['Pclass'] == 1].plot(kind='kde', sharex=True)    
data['Age'][data['Pclass'] == 2].plot(kind='kde')
data['Age'][data['Pclass'] == 3].plot(kind='kde')

plt.xlabel("Age")    
plt.title("Age Distribution within classes")
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 

df = data
fig = plt.figure(figsize=(18,4))
alpha_level = 0.65

ax1=fig.add_subplot(141)
female_highclass = df.Survived[df.Sex == 'female'][df.Pclass != 3].value_counts()
# female_highclass.plot(kind='bar', label='female highclass', color='#FA2479', alpha=alpha_level)
female_highclass.plot(kind='bar', label='female highclass', color='#FA2479', alpha=alpha_level, sharey=True)
ax1.set_xticklabels(["Survived", "Died"], rotation=0)
ax1.set_xlim(-1, len(female_highclass))
plt.title("Who Survived? with respect to Gender and Class"); plt.legend(loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
female_lowclass = df.Survived[df.Sex == 'female'][df.Pclass == 3].value_counts()
female_lowclass.plot(kind='bar', label='female, low class', color='pink', alpha=alpha_level)
ax2.set_xticklabels(["Died","Survived"], rotation=0)
ax2.set_xlim(-1, len(female_lowclass))
plt.legend(loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
male_lowclass = df.Survived[df.Sex == 'male'][df.Pclass == 3].value_counts()
male_lowclass.plot(kind='bar', label='male, low class',color='lightblue', alpha=alpha_level)
ax3.set_xticklabels(["Died","Survived"], rotation=0)
ax3.set_xlim(-1, len(male_lowclass))
plt.legend(loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
male_highclass = df.Survived[df.Sex == 'male'][df.Pclass != 3].value_counts()
male_highclass.plot(kind='bar', label='male highclass', alpha=alpha_level, color='steelblue')
ax4.set_xticklabels(["Died","Survived"], rotation=0)
ax4.set_xlim(-1, len(male_highclass))
plt.legend(loc='best')

np.random.seed(5)
x = np.arange(1, 101)
y = 20 + 3 * x + np.random.normal(0, 60, 100)

fig = plt.figure(figsize=(18,6)) 
plt.subplot(121)
plt.plot(x, y, "o")

# draw vertical line from (70,100) to (70, 250)
plt.plot([70, 70], [100, 250])

# draw diagonal line from (70, 90) to (90, 200)
plt.plot([70, 90], [90, 300])

plt.subplot(122)
plt.arrow( 0.5, 0.8, 0.0, -0.2, fc="k", ec="k", head_width=0.05, head_length=0.1)

fig, ax = plt.subplots()

ax.text(0.5, 0.5, 'hello world: $\int_0^\infty e^x dx$', size=24, ha='center', va='center')

from IPython.core.display import HTML

def css_styling():
    styles = open("custom_style.css", "r").read() #or edit path to custom.css
    return HTML(styles)
css_styling()



