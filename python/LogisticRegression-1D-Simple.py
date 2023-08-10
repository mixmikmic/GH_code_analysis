import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)  
get_ipython().magic('matplotlib inline')

def response_probability(xvec, x0):
    xvec *= 1.0                
    prob = np.zeros(len(xvec))
    prob[ xvec > x0 ] = 1.0
    return prob

# pick range for predictor variable "x"
xlo = 0.0
x0  = 3.0    # point above which response goes from 0 to 1
xhi = 10.0

# generate 50 random x-axis points
x = np.random.uniform(xlo, xhi, 50)
x = x[ x != x0 ]   # remove x0 if it happened to occur in x

# generate probabilities
response_prob = response_probability(x,x0) 

# plot the probability of the response being 0 or 1, given x
plt.rcParams['figure.figsize'] = (16,6)
plt.scatter(x,response_prob,s=50,facecolor='None')
plt.xlim(xlo,xhi)
plt.ylim(-0.1,1.1)
plt.xlabel('x (predictor)', fontsize=20)
plt.ylabel('Prob(y|x)\n',fontsize=20)
plt.tick_params(labelsize=20)

# show location of x0
plt.plot([x0,x0],[-0.1,1.1],'k--',lw=2)
plt.text(x0,-0.05,r'$x_0$', fontsize=24)

plt.show()

from sklearn.linear_model import LogisticRegression

# make a logistic regression model, keeping all the default values for the parameters
lr_model = LogisticRegression()
lr_model

# x needs to be reshaped to a column vector since it is 1-D
x = x.reshape(-1,1)

lr_model.fit(x,response_prob)

# The fitted coefficients B0 and B1 can be read-off with vars()
vars(lr_model)

def logistic_curve(x,B0,B1):
    return 1./(1.+np.exp(-(B0+B1*x)))

# plot the probability of the response being 0 or 1, given x
plt.rcParams['figure.figsize'] = (16,6)
plt.scatter(x,response_prob,s=50,facecolor='None')
plt.xlim(xlo,xhi)
plt.ylim(-0.1,1.1)
plt.xlabel('x (predictor)', fontsize=20)
plt.ylabel('Prob(y|x)\n',fontsize=20)
plt.tick_params(labelsize=20)

# show location of x0 and height of 0.5
plt.plot([x0,x0],[-0.1,1.1],'k--',lw=2)
plt.plot([xlo,xhi],[0.5,0.5],'k--',lw=2)
plt.text(x0,-0.05,r'$x_0$', fontsize=24)

# overlay fitted logistic curve
xx = np.arange(xlo,xhi,0.01)
B0 = lr_model.intercept_[0]
B1 = lr_model.coef_[0][0]
yy = logistic_curve(xx,B0,B1)
plt.plot(xx,yy,'r',lw=2)

plt.show()

# test out the model's prediction at a few new values of x
x_new = [1.0, 2.9, 8.0]
x_new = np.array(x_new).reshape(-1,1)
x_new

# the model is not perfect at classifying x values-- to the immediate left of X0 it gets it wrong:
lr_model.predict(x_new)

# re-run the model on the same data, increasing C from default = 1 to 10
lr_model = LogisticRegression(C=10)
lr_model.fit(x,response_prob)

# plot the probability of the response being 0 or 1, given x
plt.rcParams['figure.figsize'] = (16,6)
plt.scatter(x,response_prob,s=50,facecolor='None')
plt.xlim(xlo,xhi)
plt.ylim(-0.1,1.1)
plt.xlabel('x (predictor)', fontsize=20)
plt.ylabel('Prob(y|x)\n',fontsize=20)
plt.tick_params(labelsize=20)

# show location of x0 and height of 0.5
plt.plot([x0,x0],[-0.1,1.1],'k--',lw=2)
plt.plot([xlo,xhi],[0.5,0.5],'k--',lw=2)
plt.text(x0,-0.05,r'$x_0$', fontsize=24)

# overlay fitted logistic curve
xx = np.arange(xlo,xhi,0.01)
B0 = lr_model.intercept_[0]
B1 = lr_model.coef_[0][0]
yy = logistic_curve(xx,B0,B1)
plt.plot(xx,yy,'r',lw=2)

plt.show()

x_new

lr_model.predict(x_new)

