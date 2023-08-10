import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from IPython.display import Image

# read in data
list_ = []
i = 0
yr = 2006
while yr <= 2010:
    k = 0
    while k < 12:
        k += 1  
        #print ("loading: ../data/criminal_circuit_court_cases_%s/criminal_circuit_court_cases_%s_%s.csv"%(yr,yr,k))
        list_.append(pd.read_csv("../data/criminal_circuit_court_cases_%s/criminal_circuit_court_cases_%s_%s.csv"%(yr,yr,k), low_memory=False))
        i += 1  
    yr += 1  
charges_df = pd.concat(list_)

# display first 4 rows
# charges_df[:4]

# Pull out zipcode to match with census data 
charges_df['Zip'] = charges_df['Address'].str.extract('([0-9]*$)', expand=False)
# drop charges where zip can't be found
charges_df = charges_df[pd.notnull(charges_df['Zip'])]
# Make sure the zip is a number
charges_df['Zip'] = pd.to_numeric(charges_df['Zip'])
# fill in NaN (blank cells) to ease later work
charges_df = charges_df.fillna(value="")
# print new count

# Load the csv file into a dataframe
zip_df = pd.read_csv("../data/zip_income.csv") 
# There were commas in the data. So let's strip those out. 
zip_df['Median'] = zip_df['Median'].str.replace(',', '')
zip_df['Mean'] = zip_df['Mean'].str.replace(',', '')
# Also, we won't need the population column. So let's drop that too.
zip_df = zip_df.drop('Pop', 1)
# Exclude zip codes not in VA see http://www.zipcodestogo.com/Virginia/
zip_df = zip_df[zip_df['Zip']>=20101]
zip_df = zip_df[zip_df['Zip']!=23909] # note there was an error in this entry so I had to remove it
zip_df = zip_df[zip_df['Zip']<=26886]
zip_df['Mean'] = pd.to_numeric(zip_df['Mean'])
zip_df['Median'] = pd.to_numeric(zip_df['Median'])
# display first 4 rows
zip_df[:4]

# merge original data set and ACS data. THis is a join
munged_df = pd.merge(charges_df,zip_df,how='inner',on='Zip')

#
# NOTE: THE ORIGINAL VERSION OF THIS NOTEBOOK NEGLECTED TO REMOVE ENTERIES 
# WITH UNIDENTIFIED RACE AND SEX COLUMNS. THE FOLLOWING LINES CORRECT THIS.
#
munged_df = munged_df[munged_df['Race'] != '']
munged_df = munged_df[munged_df['Sex'] != '']

# Translate charge types into positions on an ordered list from 1 and 10 
munged_df['Seriousness'] = 0
munged_df.loc[(munged_df['ChargeType'].str.contains('Misdemeanor',case=False)==True) & (munged_df['Class'].str.contains('1',case=False)==True), 'Seriousness'] = 4
munged_df.loc[(munged_df['ChargeType'].str.contains('Misdemeanor',case=False)==True) & (munged_df['Class'].str.contains('2',case=False)==True), 'Seriousness'] = 3
munged_df.loc[(munged_df['ChargeType'].str.contains('Misdemeanor',case=False)==True) & (munged_df['Class'].str.contains('3',case=False)==True), 'Seriousness'] = 2
munged_df.loc[(munged_df['ChargeType'].str.contains('Misdemeanor',case=False)==True) & (munged_df['Class'].str.contains('4',case=False)==True), 'Seriousness'] = 1
munged_df.loc[(munged_df['ChargeType'].str.contains('Felony',case=False)==True) & (munged_df['Class'].str.contains('1',case=False)==True), 'Seriousness'] = 10
munged_df.loc[(munged_df['ChargeType'].str.contains('Felony',case=False)==True) & (munged_df['Class'].str.contains('2',case=False)==True), 'Seriousness'] = 9
munged_df.loc[(munged_df['ChargeType'].str.contains('Felony',case=False)==True) & (munged_df['Class'].str.contains('3',case=False)==True), 'Seriousness'] = 8
munged_df.loc[(munged_df['ChargeType'].str.contains('Felony',case=False)==True) & (munged_df['Class'].str.contains('4',case=False)==True), 'Seriousness'] = 7
munged_df.loc[(munged_df['ChargeType'].str.contains('Felony',case=False)==True) & (munged_df['Class'].str.contains('5',case=False)==True), 'Seriousness'] = 6
munged_df.loc[(munged_df['ChargeType'].str.contains('Felony',case=False)==True) & (munged_df['Class'].str.contains('6',case=False)==True), 'Seriousness'] = 5
munged_df = munged_df[munged_df['Seriousness'] > 0]

# Break out each race category so they can be considered by the linear regression
munged_df['Male'] = 0
munged_df.loc[munged_df['Sex'] == 'Male', 'Male'] = 1
munged_df['Native'] = 0
munged_df.loc[munged_df['Race'].str.contains('american',case=False)==True, 'Native'] = 1
munged_df['Asian'] = 0
munged_df.loc[munged_df['Race'].str.contains('asian',case=False)==True, 'Asian'] = 1
munged_df['Black'] = 0
munged_df.loc[munged_df['Race'].str.contains('black',case=False)==True, 'Black'] = 1
munged_df['Hispanic'] = 0
munged_df.loc[munged_df['Race'] == 'Hispanic', 'Hispanic'] = 1
munged_df['Other'] = 0
munged_df.loc[munged_df['Race'].str.contains('other',case=False)==True, 'Other'] = 1

# figure out what our sentece should be. Note: originally I was dooing more than renaming. So this is really some vestigle code. 
munged_df['SentenceDays'] = pd.to_numeric(munged_df['SentenceTimeDays'])
munged_df['SentenceDays_T'] = np.log(1+munged_df['SentenceDays'])
munged_df = munged_df.fillna(value=0)

# partition data for cross-validation
holdout = munged_df.sample(frac=0.2)
training = munged_df.loc[~munged_df.index.isin(holdout.index)]

# optional print to file
munged_df.to_csv(path_or_buf='../data/output.csv')

#output_sample = munged_df[(munged_df['Seriousness'] <= 2) & (munged_df['SentenceDays'] > 0)]
#output_sample = munged_df.sample(n=5000)
#output_sample.to_csv(path_or_buf='../data/output_sample.csv')

# display first four rows
munged_df[:4]

# Run a simple linear regression & print the P values
model = ols("SentenceDays ~ Seriousness", training).fit()
print("P-values:\n%s"%model.pvalues)

# Plot regression
fig = sns.lmplot(x="Seriousness", y="SentenceDays", data=munged_df, scatter_kws={"s": 20, "alpha": 0.25}, order=1)
plt.rc('font', family='serif', monospace='Courier') # http://matplotlib.org/users/usetex.html
plt.title("Linear Regression\n(All Data Points)",  fontsize = 17, y=1.05)
plt.xlabel('Seriousness')
plt.ylabel('Sentence in Days')
plt.annotate('R-squared: %f'%(model.rsquared), (0,0), (0,-45),  fontsize = 11, xycoords='axes fraction', textcoords='offset points', va='top')
plt.savefig('../data/tmp/f1.1.png',bbox_inches='tight'); 

# Run a simple linear regression & print the P values
model = ols("SentenceDays ~ Seriousness", training).fit()
print("P-values:\n%s"%model.pvalues)

# Plot Regression with estimator
fig = sns.lmplot(x="Seriousness", y="SentenceDays", data=munged_df, x_estimator=np.mean, order=1)
plt.rc('font', family='serif', monospace='Courier') #http://matplotlib.org/users/usetex.html
plt.title("Linear Regression\n(Representative \"Dots\")",  fontsize = 17, y=1.05)
plt.xlabel('Seriousness')
plt.ylabel('Sentence in Days')
plt.annotate('R-squared: %f'%(model.rsquared), (0,0), (0,-45), fontsize = 11, xycoords='axes fraction', textcoords='offset points', va='top')
plt.savefig('../data/tmp/f1.2.png',bbox_inches='tight'); 

# Run a simple linear regression & print the P values
model = ols("SentenceDays ~ Seriousness + np.power(Seriousness, 2)", training).fit()
print("P-values:\n%s"%model.pvalues)

# Plot multiple subplot axes with seaborn
# h/t https://gist.github.com/JohnGriffiths/8605267
fig_outfile = '../data/tmp/fig_1.3.png'

# Plot the figs and save to temp files
fig = sns.lmplot(x="Seriousness", y="SentenceDays", data=munged_df, x_estimator=np.mean, order=2);
fig = (fig.set_axis_labels("Seriousness", "Sentence in Days"))
plt.suptitle("2nd Order Polynomial Regression\n(Representative \"Dots\")",  fontsize = 13, y=1.07)
plt.annotate('R-squared: %f'%(model.rsquared), (0,0), (0,-45), fontsize = 11, xycoords='axes fraction', textcoords='offset points', va='top')
plt.savefig('../data/tmp/f1.3.1.png',bbox_inches='tight'); plt.close()
model = ols("SentenceDays ~ Seriousness + np.power(Seriousness, 2) + np.power(Seriousness, 3)+ np.power(Seriousness, 4)", munged_df).fit()
print("P values:\n%s"%model.pvalues)
fig = sns.lmplot(x="Seriousness", y="SentenceDays", data=munged_df, x_estimator=np.mean, order=4);
fig = (fig.set_axis_labels("Seriousness", "Sentence in Days"))
plt.suptitle("4th Order Polynomial Regression\n(Representative \"Dots\")",  fontsize = 13, y=1.07)
plt.annotate('R-squared: %f'%(model.rsquared), (0,0), (0,-45), fontsize = 11, xycoords='axes fraction', textcoords='offset points', va='top')
plt.savefig('../data/tmp/f1.3.2.png',bbox_inches='tight'); plt.close()

# Combine them with imshows
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
for a in [1,2]: ax[a-1].imshow(plt.imread('../data/tmp/f1.3.%s.png' %a)); ax[a-1].axis('off')
plt.tight_layout(); plt.savefig(fig_outfile,bbox_inches='tight'); plt.close() 

# Display in notebook as an image
Image(fig_outfile, width="100%")

# Run a simple linear regression & print the P values
model = ols("SentenceDays_T ~ Seriousness", training).fit()
print("P-values:\n%s"%model.pvalues)

# Plot multiple subplot axes with seaborn
# h/t https://gist.github.com/JohnGriffiths/8605267
fig_outfile = '../data/tmp/fig_2.png'

# Plot the figs and save to temp files
fig = sns.lmplot(x="Seriousness", y="SentenceDays_T", data=munged_df, scatter_kws={"s": 20, "alpha": 0.25}, order=1);
fig = (fig.set_axis_labels("Seriousness", "log(1 + Sentence in Days)"))
plt.suptitle("(All Data Points)",  fontsize = 14, y=1.03)
plt.savefig('../data/tmp/f2.1.png',bbox_inches='tight'); plt.close()
fig = sns.lmplot(x="Seriousness", y="SentenceDays_T", data=munged_df, x_estimator=np.mean, order=1);
fig = (fig.set_axis_labels("Seriousness", "log(1 + Sentence in Days)"))
plt.suptitle("(Representative \"Dots\")",  fontsize = 14, y=1.03)
plt.savefig('../data/tmp/f2.2.png',bbox_inches='tight'); plt.close()

# Combine them with imshows
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
for a in [1,2]: ax[a-1].imshow(plt.imread('../data/tmp/f2.%s.png' %a)); ax[a-1].axis('off')
plt.suptitle("Log-Linear Regression",  fontsize = 17, y=1.02)
plt.tight_layout();
plt.annotate('R-squared: %f'%(model.rsquared), (-1.05,0), (0,-10), fontsize = 11, xycoords='axes fraction', textcoords='offset points', va='top')
plt.savefig(fig_outfile,bbox_inches='tight'); plt.close() 

# Display in notebook as an image
Image(fig_outfile, width="100%")

# Run a simple linear regression & print the P values
model = ols("SentenceDays_T ~ Seriousness + np.power(Seriousness, 2)", training).fit()
print("P-values:\n%s"%model.pvalues)

# Plot multiple subplot axes with seaborn
# h/t https://gist.github.com/JohnGriffiths/8605267
fig_outfile = '../data/tmp/fig_3.png'

# Plot the figs and save to temp files
fig = sns.lmplot(x="Seriousness", y="SentenceDays_T", data=munged_df, x_estimator=np.mean, order=2);
fig = (fig.set_axis_labels("Seriousness", "log(1 + Sentence in Days)"))
plt.suptitle("2nd Order Polynomial Regression\n(Representative \"Dots\")",  fontsize = 13, y=1.07)
plt.annotate('R-squared: %f'%(model.rsquared), (0,0), (0,-45),  fontsize = 11, xycoords='axes fraction', textcoords='offset points', va='top')
plt.savefig('../data/tmp/f3.1.png',bbox_inches='tight'); plt.close()
model = ols("SentenceDays_T ~ Seriousness + np.power(Seriousness, 2) + np.power(Seriousness, 3)+ np.power(Seriousness, 4)", munged_df).fit()
fig = sns.lmplot(x="Seriousness", y="SentenceDays_T", data=munged_df, x_estimator=np.mean, order=4);
fig = (fig.set_axis_labels("Seriousness", "log(1 + Sentence in Days)"))
plt.suptitle("4th Order Polynomial Regression\n(Representative \"Dots\")",  fontsize = 13, y=1.07)
plt.annotate('R-squared: %f'%(model.rsquared), (0,0), (0,-45),  fontsize = 11, xycoords='axes fraction', textcoords='offset points', va='top')
plt.savefig('../data/tmp/f3.2.png',bbox_inches='tight'); plt.close()

# Combine them with imshows
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
for a in [1,2]: ax[a-1].imshow(plt.imread('../data/tmp/f3.%s.png' %a)); ax[a-1].axis('off')
plt.tight_layout(); plt.savefig(fig_outfile,bbox_inches='tight'); plt.close() 

# Display in notebook as an image
Image(fig_outfile, width="100%")

# Plot multiple linear regression 
# h/t https://www.datarobot.com/blog/multiple-regression-using-statsmodels/#appendix

from mpl_toolkits.mplot3d import Axes3D

X = training[['Seriousness', 'Mean']]
y = training['SentenceDays_T']

## fit a OLS model
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()

## Create the 3d plot 
xx1, xx2 = np.meshgrid(np.linspace(X.Seriousness.min(), X.Seriousness.max(), 100), 
                       np.linspace(X.Mean.min(), X.Mean.max(), 100))

# plot the hyperplane by evaluating the parameters on the grid
Z = est.params[0] + est.params[1] * xx1 + est.params[2] * xx2

# create matplotlib 3d axes
fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig, azim=-120, elev=15)

# plot hyperplane
surf = ax.plot_surface(xx1, xx2, Z, cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0)

# plot data points
resid = y - est.predict(X)
ax.scatter(X[resid >= 0].Seriousness, X[resid >= 0].Mean, y[resid >= 0], color='black', alpha=0.25, facecolor='white')
ax.scatter(X[resid < 0].Seriousness, X[resid < 0].Mean, y[resid < 0], color='black', alpha=0.25)

# set axis labels
ax.set_title('Multiple Log-Linear Regression', fontsize = 20)
ax.set_xlabel('Seriousness')
ax.set_ylabel('Mean Income')
ax.set_zlabel('log(1 + Sentence in Days)')

print("========================")
print("         LINEAR")
print("========================")
model = ols("SentenceDays ~ Seriousness + Male + Mean + Black + Hispanic + Asian + Native + Other", training).fit()
print(model.summary())
plt.scatter(model.fittedvalues,model.resid,alpha=0.25)
plt.xlabel('perdicted')
plt.ylabel('residuals')
plt.show()
model = ols("SentenceDays ~ Seriousness + Male + Mean + Black + Hispanic + Asian + Native + Other", holdout).fit()
print(model.summary())
plt.scatter(model.fittedvalues,model.resid,alpha=0.25)
plt.xlabel('perdicted')
plt.ylabel('residuals')
plt.show()

print("========================")
print("       2ND ORDER")
print("========================")
model = ols("SentenceDays ~ Seriousness + np.power(Seriousness, 2) + Male + Mean + Black + Hispanic + Asian + Native + Other", training).fit()
print(model.summary())
plt.scatter(model.fittedvalues,model.resid,alpha=0.25)
plt.xlabel('perdicted')
plt.ylabel('residuals')
plt.show()
model = ols("SentenceDays ~ Seriousness + np.power(Seriousness, 2) + Male + Mean + Black + Hispanic + Asian + Native + Other", holdout).fit()
print(model.summary())
plt.scatter(model.fittedvalues,model.resid,alpha=0.25)
plt.xlabel('perdicted')
plt.ylabel('residuals')
plt.show()

print("========================")
print("       4TH ORDER")
print("========================")
model = ols("SentenceDays ~ Seriousness + np.power(Seriousness, 2) + Male + Mean + Black + Hispanic + Asian + Native + Other", training).fit()
print(model.summary())
plt.scatter(model.fittedvalues,model.resid,alpha=0.25)
plt.xlabel('perdicted')
plt.ylabel('residuals')
plt.show()
model = ols("SentenceDays ~ Seriousness + np.power(Seriousness, 2) + Male + Mean + Black + Hispanic + Asian + Native + Other", holdout).fit()
print(model.summary())
plt.scatter(model.fittedvalues,model.resid,alpha=0.25)
plt.xlabel('perdicted')
plt.ylabel('residuals')
plt.show()

print("========================")
print("      LOG-LINEAR")
print("========================")
model = ols("np.log(1+SentenceDays) ~ Seriousness + Male + Mean + Black + Hispanic + Asian + Native + Other", training).fit()
print(model.summary())
plt.scatter(model.fittedvalues,model.resid,alpha=0.25)
plt.xlabel('perdicted')
plt.ylabel('residuals')
plt.show()
model = ols("np.log(1+SentenceDays) ~ Seriousness + Male + Mean + Black + Hispanic + Asian + Native + Other", holdout).fit()
print(model.summary())
plt.scatter(model.fittedvalues,model.resid,alpha=0.25)
plt.xlabel('perdicted')
plt.ylabel('residuals')
plt.show()

print("========================")
print("  2ND ORDER LOG-LINEAR")
print("========================")
model = ols("np.log(1+SentenceDays) ~ Seriousness + np.power(Seriousness, 2) + Male + Mean + Black + Hispanic + Asian + Native + Other", training).fit()
print(model.summary())
plt.scatter(model.fittedvalues,model.resid,alpha=0.25)
plt.xlabel('perdicted')
plt.ylabel('residuals')
plt.show()
model = ols("np.log(1+SentenceDays) ~ Seriousness + np.power(Seriousness, 2) + Male + Mean + Black + Hispanic + Asian + Native + Other", holdout).fit()
print(model.summary())
plt.scatter(model.fittedvalues,model.resid,alpha=0.25)
plt.xlabel('perdicted')
plt.ylabel('residuals')
plt.show()

print("========================")
print("  4TH ORDER LOG-LINEAR")
print("========================")
model = ols("np.log(1+SentenceDays) ~ Seriousness + np.power(Seriousness, 2) + Male + Mean + Black + Hispanic + Asian + Native + Other", training).fit()
print(model.summary())
plt.scatter(model.fittedvalues,model.resid,alpha=0.25)
plt.xlabel('perdicted')
plt.ylabel('residuals')
plt.show()
model = ols("np.log(1+SentenceDays) ~ Seriousness + np.power(Seriousness, 2) + Male + Mean + Black + Hispanic + Asian + Native + Other", holdout).fit()
print(model.summary())
plt.scatter(model.fittedvalues,model.resid,alpha=0.25)
plt.xlabel('perdicted')
plt.ylabel('residuals')
plt.show()

model = ols("np.log(1+SentenceDays) ~ Seriousness + Male + Mean + Black + Hispanic + Asian + Native + Other", munged_df).fit()
#model = ols("np.log(1+SentenceDays) ~ Seriousness + Male + Mean + C(Race,Treatment(reference='White Caucasian (Non-Hispanic)'))", munged_df).fit()
model.summary()



