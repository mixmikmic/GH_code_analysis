#Import Basic Packages
import numpy as np
import pandas as pd
import sqlite3
from pandas import Series,DataFrame

# These are the plotting modules and libraries we'll use:
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Import for Linear Regression
import sklearn
from sklearn.linear_model import LinearRegression

# Command so that plots appear in the iPython Notebook
get_ipython().magic('matplotlib inline')

#Establish Connection to SQLite Database
con = sqlite3.connect('Documents/Python/world-development-indicators/database.sqlite')

#Define SQL Query
#We're really only interestd in recent history, so for now we'll focus on data beyond and including the year 2000.
Indicators = '''

SELECT I.CountryName, C.Region, C.IncomeGroup, I.IndicatorName, I.Year, I.Value
FROM INDICATORS I
LEFT JOIN COUNTRY C on I.CountryCode = C.CountryCode
WHERE YEAR >= 2000
AND  IndicatorName in 
(
 'Alternative and nuclear energy (% of total energy use)'
,'Energy use (kg of oil equivalent per capita)'
,'GDP per capita, PPP (current international $)'
,'Improved water source (% of population with access)'
,'Labor force with tertiary education (% of total)'
,'Life expectancy at birth, total (years)'
,'Internet users (per 100 people)'
)

'''

#Create Dataframe and show sample
Indicators_Raw = pd.read_sql(Indicators,con)
Indicators_Raw.head()

Indicators_Raw.drop(Indicators_Raw[Indicators_Raw.Year >= 2014].index, inplace = True)

Indicators_Raw.drop(Indicators_Raw[Indicators_Raw.Region == ''].index, inplace = True)

#Convert Indicators into Columns
Indicators_df = Indicators_Raw.set_index(
    ['Year', 'Region', 'IncomeGroup', 'CountryName', 'IndicatorName']).unstack('IndicatorName')


#Reformat Columns
Indicators_df.reset_index(level =  ['Year', 'Region', 'IncomeGroup', 'CountryName'], inplace = True)

Indicators_df.columns = [' '.join(col).strip() for col in Indicators_df.columns.values]
Indicators_df.columns = [col.strip('Value ') if col not in ('Year', 'CountryName') else col for col in Indicators_df.columns]
Indicators_df.head(5)

Indicatorsnn_df = Indicators_df.dropna()

Indicators_df.groupby(['Year']).mean().pct_change()

def ExploreData(group):
    df = Indicatorsnn_df
    columns = df.groupby(['Year']).mean().columns
    var = Series.unique(df[group])
    outercount = 0
    innercount = 0
    titlesize = 35
    axeslabelsize = 40
    ticksize = 28
    legendsize = 25
    fig, ax = plt.subplots(len(columns),2,figsize=(40,130))
    for i in range(len(columns)):
        ax[i,0].plot(df.groupby(['Year']).mean().index, 
                df.groupby(['Year']).mean()[columns[outercount]], 
                    linewidth = 7,
                    label = 'Total')
        ax[i,1].hist(df[columns[outercount]],
                    alpha = .15, 
                    bins = 20,
                    label = 'Total')
        plt.subplots_adjust(hspace = .5)
        ax[i,0].set_title(columns[outercount] + '\n', fontsize = titlesize, fontweight = 'bold')
        ax[i,1].set_title(columns[outercount] + '\n', fontsize = titlesize, fontweight = 'bold')
        ax[i,0].set_xlabel('\n Year', fontsize = axeslabelsize)
        ax[i,1].set_xlabel('\n Value', fontsize = axeslabelsize)
        ax[i,1].set_ylabel('Count', fontsize = axeslabelsize)
        ax[i,0].tick_params(labelsize=ticksize)
        ax[i,1].tick_params(labelsize=ticksize)
        for y in var:
            ax[i,0].plot(df[df[group] == y]
                    .groupby(['Year']).mean()[columns[outercount]]
                    ,linewidth = 4 
                    ,label = y)
            ax[i,0].legend(loc='upper left', shadow=True, prop= {'size':legendsize})
            ax[i,1].hist(df[df[group]==y][columns[outercount]],
                        alpha = .40,
                        bins = 20,
                        label = y)
            ax[i,1].plot
            ax[i,1].legend(loc='upper left', shadow=True, prop= {'size':legendsize})
            innercount = innercount +1
        outercount = outercount + 1

ExploreData('IncomeGroup')

ExploreData('Region')

Indicators_df.drop('Year',axis = 1).corr(method = 'pearson') 

sns.set(font_scale=1.3)

sns.pairplot(Indicators_df[Indicators_df['Year'] == 2012]
             ,x_vars = ['GDP per capita, PPP (current international $)'
             ,'Alternative and nuclear energy (% of total energy use)']
             ,y_vars = ['Energy use (kg of oil equivalent per capita)']
             ,size = 7
             ,hue = 'Region')

sns.set(font_scale=1.3)

sns.pairplot(Indicators_df[Indicators_df['Year'] == 2012]
             ,x_vars = ['GDP per capita, PPP (current international $)'
             ,'Alternative and nuclear energy (% of total energy use)']
             ,y_vars = ['Energy use (kg of oil equivalent per capita)']
             ,size = 7
             ,hue = 'IncomeGroup')

#Create a linear regression object
lreg = LinearRegression()


X_cols = Indicatorsnn_df.drop(['Year'
                               ,'Region' 
                               ,'CountryName'
                               ,'Energy use (kg of oil equivalent per capita)'
                               ,'IncomeGroup']
                              ,1)
Y_target = Indicatorsnn_df['Energy use (kg of oil equivalent per capita)']

#Implement Linear Regression
lreg.fit(X_cols, Y_target)

#Create a dataframe from the features
coeff_df = DataFrame(X_cols.columns)
coeff_df.columns = ['Indicators']

coeff_df["Coefficient Estimate"] = pd.Series(lreg.coef_)

coeff_df

# Grab the output and set as X and Y test and train data sets
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X_cols,Y_target)

# Create our regression object
lreg = LinearRegression()

# Once again do a linear regression, except only on the training sets this time
lreg.fit(X_train,Y_train)

# Predictions on training and testing sets
pred_train = lreg.predict(X_train)
pred_test = lreg.predict(X_test)

#Resize for imporoved visibility
plt.figure(figsize = (20,10))

# Scatter plot the training data
train = plt.scatter(pred_train,(pred_train-Y_train),c='b',alpha=0.5)

# Scatter plot the testing data
test = plt.scatter(pred_test,(pred_test-Y_test),c='r',alpha=0.5)


#Labels
plt.legend((train,test),('Training','Test'),loc='lower left')
plt.title('Residual Plots')
plt.xlabel('Energy Usage Per Capita')




