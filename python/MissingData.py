import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
#Missingno is a package to visualize missing data
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')
#Load data
df = pd.read_csv('nba_2016_2017_100.csv')

display(df.head())
df.describe()

msno.matrix(df)

msno.bar(df, color="blue", figsize=(30,18))

msno.heatmap(df, figsize=(5,5))

df_new = df.dropna()
display(df_new.describe())
display(df.describe())

df['GP'].hist()
df_new['GP'].hist()
plt.title('Games Played')
plt.xlabel('Number of Games Played')
plt.ylabel('Count')
plt.show()

impute = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
salary = df['SALARY_MILLIONS'].values
gp = df['GP'].values
salary_mean = impute.fit_transform(salary.reshape(-1,1)).reshape(1,-1)[0]
plt.scatter(gp,salary_mean,c='r')
plt.scatter(gp,salary,c='b')
plt.xlabel('Games Played')
plt.ylabel('Salary ($ Millions)')
plt.title('Imputation Using Mean')
plt.show()

linear_model = LinearRegression()
result = linear_model.fit(df_new['GP'].values.reshape(-1,1),df_new['SALARY_MILLIONS'].values)
xvals = np.arange(0,90,5)
missing_salaries_GP = df[df['SALARY_MILLIONS'].isnull()]['GP'].values
imputed_salaries = result.predict(missing_salaries_GP.reshape(-1,1))
plt.plot(xvals,result.predict(xvals.reshape(-1,1)),'k--')
plt.scatter(df['GP'].values,df['SALARY_MILLIONS'].values,c='b')
plt.scatter(missing_salaries_GP,imputed_salaries,c='r')
plt.xlabel('Games Played')
plt.ylabel('Salary ($ Millions)')
plt.title('Imputation using Regression')
plt.show()

#Calculate residuals
residuals = df_new['SALARY_MILLIONS'].values-result.predict(df_new['GP'].values.reshape(-1,1))
var_fit = np.std(residuals)
#Add noise
imputed_salaries_noise = []
for item in imputed_salaries:
    imputed_salaries_noise.append(max(0,np.random.normal(loc=item,scale=var_fit)))
plt.plot(xvals,result.predict(xvals.reshape(-1,1)),'k--')
plt.scatter(missing_salaries_GP,imputed_salaries_noise,c='r')
plt.scatter(df['GP'].values,df['SALARY_MILLIONS'].values,c='b')
#plt.scatter(missing_salaries_GP,imputed_salaries_noise,c='r')
plt.xlabel('Games Played')
plt.ylabel('Salary ($ Millions)')
plt.title('Imputation using Regression')
plt.show()

#Turn defensive ratings into class labels
df_class = df[['OFF_RATING','DEF_RATING','PTS']].dropna()
offensive_rating = df_class['OFF_RATING'].values
defensive_rating = df_class['DEF_RATING'].values
points = df_class['PTS'].values
defensive_class = []
for i in range(len(offensive_rating)):
    if defensive_rating[i] > 105:
        defensive_class.append(1)
    elif defensive_rating[i] <= 105:
        defensive_class.append(0)

classifier = KNeighborsClassifier(n_neighbors=3)
class_result = classifier.fit(np.stack((offensive_rating,points),axis=1),defensive_class)
#Find missing data
missing_def_rating = df[df['DEF_RATING'].isnull()][['OFF_RATING','PTS']].values
#Predict
predicted_def_rating = class_result.predict(missing_def_rating)
plt.scatter(missing_def_rating[:,0],missing_def_rating[:,1],c=predicted_def_rating,marker='s')
plt.scatter(offensive_rating,points,c=defensive_class)
plt.title('Defensive Top Performers')
plt.xlabel('Offensive Rating')
plt.ylabel('Points')
plt.show()

from statsmodels.imputation import mice
import statsmodels as sm
#Create mice data instance
imp = mice.MICEData(df[['GP','SALARY_MILLIONS']])
#Linear model
fml = 'SALARY_MILLIONS ~ GP'
#Build MICE pipeline
out = mice.MICE(fml, sm.regression.linear_model.OLS, imp)
#Fit with burn in of 3 and 10 imputations
results = out.fit(3,10)
#Output results
display(results.summary())
#plot
imp.plot_bivariate('GP','SALARY_MILLIONS')



