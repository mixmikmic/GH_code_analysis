import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

get_ipython().run_line_magic('matplotlib', 'inline')

default = pd.read_csv('../../data/Default.csv', index_col=0)
print('{}\n\n'.format(default.describe()))
print(default.describe(include=['O']))

fig = plt.figure(figsize=(12, 5))

#this will create axes in which the left plot spans 2 columns 
#and the right two span only 1 column
ax1 = plt.subplot(121)
ax2 = plt.subplot(143)
ax3 = plt.subplot(144)

#take 1000 samples where the individual did not default
default_no_sample = default[default['default'] == 'No'].sample(n=1000)
ax1.scatter(default_no_sample['balance'].values,
            default_no_sample['income'].values,
            marker='o', edgecolors='blue', facecolors='None', s=25)
#include all samples where the individual did default
ax1.scatter(default[default['default'] == 'Yes']['balance'].values,
            default[default['default'] == 'Yes']['income'].values,
            marker='+', color='orange')
ax1.set(xlabel='Balance', ylabel='Income')

sns.boxplot(x=default['default'], y=default['balance'], ax=ax2)

sns.boxplot(x=default['default'], y=default['income'], ax=ax3)

fig.tight_layout()

#figure 4.2

#create dummy variables for default and student
default2 = pd.get_dummies(default, drop_first=True)
#change the unites of income from $1 to $1000
default2['income'] = default2['income'] / 1000
#create linear regression and logit regression models
default_ols = smf.ols('default_Yes ~ balance', default2).fit()
default_logit = smf.logit('default_Yes ~ balance', default2).fit()
#create x variable for predictions
x = np.linspace(0, 2700, 1000)
#make predictions for each model
default_ols_pred = default_ols.predict(exog={'balance': x});
default_logit_pred = default_logit.predict(exog={'balance': x});
#create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
#left panel
ax1.scatter(default2['balance'], default2['default_Yes'], marker='|', color='orange')
ax1.hlines(0, -100, 3000, linestyles='dashed', lw=1)
ax1.hlines(1, -100, 3000, linestyles='dashed', lw=1)
ax1.set(xlim=[min(x)-50, max(x)], xlabel='Balance', ylabel='Probability of Default',
        title='Linear Regression');
#this plots the linear regression line
ax1.plot(x, default_ols_pred);
#right panel
ax2.scatter(default2['balance'], default2['default_Yes'], marker='|', color='orange')
ax2.hlines(0, -100, 3000, linestyles='dashed', lw=1)
ax2.hlines(1, -100, 3000, linestyles='dashed', lw=1)
ax2.set(xlim=[min(x)-50, max(x)], xlabel='Balance', ylabel='Probability of Default',
        title='Linear Regression');
#this plots the logistic regression line
ax2.plot(x, default_logit_pred);

#table 4.1
default_logit.summary().tables[1]

default_logit = smf.logit('default_Yes ~ student_Yes', default2).fit()
default_logit.summary().tables[1]

default_logit = smf.logit('default_Yes ~ balance + income + student_Yes', default2).fit()
default_logit.summary().tables[1]

#create dataframes for students and non-students
default_student = default2[['default_Yes', 'student_Yes', 'balance']]
student_yes = default_student[default_student['student_Yes'] == 1]
student_no = default_student[default_student['student_Yes'] == 0]

#create the logistic regression for students and non-students
student_yes_logit = smf.logit('default_Yes ~ balance', student_yes).fit()
student_no_logit = smf.logit('default_Yes ~ balance', student_no).fit()

#create a range of balance for predicting
x_test = np.arange(default2['balance'].min(), default2['balance'].max())

#make predictions for students and non-students
y_pred_student_yes = student_yes_logit.predict(exog={'balance': x_test})
y_pred_student_no = student_no_logit.predict(exog={'balance': x_test})

#calculate the total default percentage for students and non-students
student_yes_default = student_yes['default_Yes'].sum() / len(student_yes)
student_no_default = student_no['default_Yes'].sum() / len(student_no)

#figure 4.3

#create the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

#plot the default rate for students and non-students
y_pred_student_yes.plot(color='orange', ax=ax1)
y_pred_student_no.plot(color='lightblue', ax=ax1)

#left panel
#plot the total default rate for students and non-students
ax1.hlines(student_yes_default, 0, 2700, linestyles='--', 
           lw=0.75, color='orange')
ax1.hlines(student_no_default, 0, 2700, linestyles='--',
           lw=0.75, color='lightblue')
ax1.set(xlim=[500, 2500], xlabel='Credit Card Balance', 
        ylabel='Default Rate')
ax1.legend(['Student', 'Non-student'], loc=0)

#right panel
#make the boxplot
bp = sns.boxplot(x=default2['student_Yes'], y=default2['balance'], ax=ax2)
#change the facecolors to match the left panel
bp.artists[1].set_facecolor('orange')
bp.artists[0].set_facecolor('lightblue')
#change the xtick labels (from 0 and 1)
ax2.set(xticklabels=['No', 'Yes'], xlabel='Student Status', 
        ylabel='Credit Card Balance');

#figure 4.4
from scipy.stats import norm

mu_1 = -1.25
mu_2 = 1.25
variance = 1
sigma = np.sqrt(variance)
x_1 = np.linspace(mu_1 - 4*sigma, mu_1 + 4*sigma, 100)
x_2 = np.linspace(mu_2 - 4*sigma, mu_2 + 4*sigma, 100)
y_1 = norm.pdf(x_1, loc=mu_1, scale=1)
y_2 = norm.pdf(x_2, loc=mu_2, scale=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(x_1, y_1, color='green')
ax1.plot(x_2, y_2, color='pink')
ax1.vlines(0, -1, .5, linestyles='--', color='k')
ax1.set(ylim=[-.05,0.5])

x_1_sample = np.random.normal(loc=mu_1, scale=sigma, size=20)
x_2_sample = np.random.normal(loc=mu_2, scale=sigma, size=20)
x = (x_1_sample.mean() + x_2_sample.mean())/2
ax2.hist(x_1_sample, color='green', edgecolor='darkgreen')
ax2.hist(x_2_sample, color='pink', alpha=0.5, edgecolor='red')
ax2.vlines(0, 0, 6, linestyles='--', color='k', lw=1)
ax2.vlines(x, 0, 6, linestyles='--', color='r', lw=1)
ax2.set(ylim=[0,5]);
ax2.legend(['Bayes Decision Boundary', 'LDA Decision Boundary'])

bayes_error = []
lda_error = []
#this is already calculated above
#x = (x_1_sample.mean() + x_2_sample.mean())/2
for i in range(10000):
    x_1_sample = np.random.normal(loc=mu_1, scale=sigma, size=20)
    x_2_sample = np.random.normal(loc=mu_2, scale=sigma, size=20)
    bayes_error.append((x_1_sample > 0).sum()/20)
    bayes_error.append((x_2_sample < 0).sum()/20)
    lda_error.append((x_1_sample > x).sum()/20)
    lda_error.append((x_2_sample < x).sum()/20)
print('Bayes Error Rate: {:.2f}%'.format(np.mean(bayes_error)*100))
print('LDA Error Rate: {:.2f}%'.format(np.mean(lda_error)*100))

#figure 4.6

from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#define the mu_k's
c1_means = [.5, 2.5]
c2_means = [2.5, .5]
c3_means = [-2, -2]
#define the covariance matrix
cov = [[1, 0.5], [0.5, 1]]
#create the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
#calculate the eigen vectors (for width and height of ellipse)
vals, vecs = np.linalg.eig(cov)
order = vals.argsort()[::-1]
vals, vecs = vals[order], vecs[:, order]
#calculate the angle for the ellipse
theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
#get the width and the height (3 standard deviations is about 95% confidence)
width, height = 2 * 3 *np.sqrt(vals)
#add each ellipse to the plot
for cl, col in zip([c1_means, c2_means, c3_means],['orange', 'blue', 'green']):
    ellip = Ellipse(xy=cl, width=width, height=height, angle=theta, edgecolor=col, 
                    facecolor='none', lw=2)
    ax1.add_artist(ellip)
#change the x-limits
ax1.set(xlim=[-6, 6], ylim=[-6, 6]);
#create x1's and x2's for three different classes
c1_x1, c1_x2 = np.random.multivariate_normal(c1_means, cov, 100000).T
c2_x1, c2_x2 = np.random.multivariate_normal(c2_means, cov, 100000).T
c3_x1, c3_x2 = np.random.multivariate_normal(c3_means, cov, 100000).T
#this section will create the dataframe for modeling
x1 = np.concatenate((c1_x1, c2_x1, c3_x1))
x2 = np.concatenate((c1_x2, c2_x2, c3_x2))
#assign each point to a class (0, 1, 2)
class0 = np.zeros_like(c1_x1)
class1 = np.ones_like(c1_x1)
class2 = np.ones_like(c1_x1)*2
classes = np.concatenate((class0, class1, class2))
#make the dataframe
df = pd.DataFrame({'x1': x1, 'x2': x2, 'class': classes})
#cast the class as an integer
df['class'] = df['class'].astype('int')
#create np arrays for the predictors and response
X = df[['x1', 'x2']].values
y = df['class'].values
#create an instance of the LDA model
lda=LDA()
#fit the model to the data
lda.fit(X, y)
#create a mesh grid for plotting
X1, X2 = np.meshgrid(np.linspace(-6, 6, 1000), np.linspace(-6, 6, 1000))
#now add contours based on predictions for each point in the grid
ax1.contourf(X1, X2, lda.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
                 alpha=0.2, cmap=ListedColormap(('orange', 'blue', 'green')))
ax1.contour(X1, X2, lda.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            linestyles='dashed', colors='black', linewidths=0.4)
ax2.contourf(X1, X2, lda.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
                 alpha=0.2, cmap=ListedColormap(('orange', 'blue', 'green')))
ax2.contour(X1, X2, lda.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            linestyles='dashed', colors='black', linewidths=0.3)
#now create the samples plot
c1_x1_sample, c1_x2_sample = np.random.multivariate_normal(c1_means, cov, 20).T
c2_x1_sample, c2_x2_sample = np.random.multivariate_normal(c2_means, cov, 20).T
c3_x1_sample, c3_x2_sample = np.random.multivariate_normal(c3_means, cov, 20).T
class_0_sample = np.zeros_like(c1_x1_sample)
class_1_sample = np.ones_like(c1_x1_sample)
class_2_sample = class_1_sample * 2
x1_sample = np.concatenate((c1_x1_sample, c2_x1_sample, c3_x1_sample))
x2_sample = np.concatenate((c1_x2_sample, c2_x2_sample, c3_x2_sample))
classes_sample = np.concatenate((class_0_sample, class_1_sample, class_2_sample))
df_sample = pd.DataFrame({'x1': x1_sample, 'x2': x2_sample, 'class': classes_sample})
df['class'] = df['class'].astype('int')
X_sample = df_sample[['x1', 'x2']].values
y_sample = df_sample['class'].values
lda_sample = LDA()
lda_sample.fit(X_sample, y_sample)
ax2.scatter(c1_x1_sample, c1_x2_sample, marker='.', color='orange')
ax2.scatter(c2_x1_sample, c2_x2_sample, marker='.', color='blue')
ax2.scatter(c3_x1_sample, c3_x2_sample, marker='.', color='green')
ax2.contour(X1, X2, lda_sample.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
            colors='red', linestyles='dotted', linewidths=0.5)

ax1.set(xlabel='X1', ylabel='X2')
ax2.set(xlabel='X1', ylabel='X2')

#Table 4.4

from sklearn.metrics import confusion_matrix

default = pd.read_csv('../../data/Default.csv', index_col=0)
default = pd.get_dummies(default, drop_first=True)

X = default[['balance', 'student_Yes']].values
y = default['default_Yes'].values

lda = LDA()
lda.fit(X, y)
y_pred = lda.predict(X)
cm = confusion_matrix(y, y_pred)
cm.T

#Table 4.5

y_prob = lda.predict_proba(X)
y_prob[y_prob >= 0.2] = 1
y_prob[y_prob < 0.2] = 0
cm = confusion_matrix(y, y_prob[:, 1])
cm.T

#figure 4.7

thresh = np.linspace(0, 0.5, 100)
total_error_rate = []
default_incorrect = []
non_default_incorrect = []
for t in thresh:
    y_prob = lda.predict_proba(X)
    y_prob[y_prob >= t] = 1
    y_prob[y_prob < t] = 0
    cm = confusion_matrix(y, y_prob[:, 1])
    total_error_rate.append((cm[0, 1] + cm[1, 0]) / cm.sum())
    default_incorrect.append(cm[1, 0] / cm[1, :].sum())
    non_default_incorrect.append(cm[0, 1] / cm[0, :].sum())
    
fig, ax = plt.subplots(1, 1, figsize=(12, 4))

ax.plot(thresh, total_error_rate, color='k')
ax.plot(thresh, default_incorrect, color='b', linestyle='--')
ax.plot(thresh, non_default_incorrect, color='orange', linestyle='dotted', lw=3)
ax.set(xlabel='Threshold', ylabel='Error Rate')

ax.legend(['Total Error Rate', 'Default Error Rate', 'Non-default Error Rate'])

#figure 4.8

from sklearn.metrics import roc_curve, auc

y_prob = lda.predict_proba(X)
thresh = np.linspace(0, 1, 100)
tpr = []
fpr = []
for t in thresh:
    yp = y_prob.copy()
    yp[yp >= t] = 1
    yp[yp < t] = 0
    cm = confusion_matrix(y, yp[:, 1])
    tpr.append(cm[1, 1] / cm[1, :].sum())
    fpr.append(1 - cm[0, 0] / cm[0, :].sum())

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1])
ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')

print('AUC: {:.4f}'.format(auc(fpr, tpr)))

#Figure 4.9 right panel

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

c1_means = [-1, -1]
cov1 = [[1, 0.7], [0.7, 1]]
c1_x1, c1_x2 = np.random.multivariate_normal(c1_means, cov1, 100).T
c1_class = np.zeros_like(c1_x1)
c2_means = [2, 2]
cov2 = [[1, -.7], [-.7, 1]]
c2_x1, c2_x2 = np.random.multivariate_normal(c2_means, cov2, 100).T
c2_class = np.ones_like(c2_x1)

df = pd.DataFrame({'x1': np.concatenate((c1_x1, c2_x1)),
                   'x2': np.concatenate((c1_x2, c2_x2)),
                   'class': np.concatenate((c1_class, c2_class))})
df['class'] = df['class'].astype('int')
X = df[['x1', 'x2']].values
y = df['class'].values

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

for cov, col, cmean in zip([cov1, cov2], ['orange', 'blue'], [c1_means, c2_means]):
    vals, vecs = np.linalg.eig(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * 3 * np.sqrt(vals)
    ellip = Ellipse(xy=cmean, width=width, height=height, angle=theta, edgecolor=col,
                   facecolor='none', lw=2)
    ax1.add_artist(ellip)
ax1.scatter(c1_x1, c1_x2, marker='.', color='orange')
ax1.scatter(c2_x1, c2_x2, marker='.', color='blue')
ax1.set(xlim=[-5, 6], ylim=[-5, 6])

lda = LDA()
qda = QDA()

lda.fit(X, y)
qda.fit(X, y)

X1, X2 = np.meshgrid(np.linspace(-6, 6, 1000), np.linspace(-6, 6, 1000))

ax1.contourf(X1, X2, qda.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha=0.2, cmap=ListedColormap(('orange', 'blue')))
ax1.contour(X1, X2, qda.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
            colors='green', liewidths=2)
ax1.contour(X1, X2, lda.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            colors='black', linestyles='dotted', linewidths=0.5);
ax1.set(xlabel=r'$X_1$', ylabel=r'$X_2$');



