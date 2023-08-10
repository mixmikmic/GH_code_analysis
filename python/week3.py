#silence some warnings
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import statsmodels.api as sm

#With the inline backend, these need to be in the same cell.
fig = plt.figure(figsize=(6,3))  #Create a new empty figure object. Size is optional.
ax1 = fig.add_subplot(121)  #Layout: (1x2) axes. Add one in row 1, column 1, and make it current (what plt.* commands operate on).
ax2 = fig.add_subplot(122)  #Add an axes in row 1, column 2, and make it current. 
ax1.plot(range(10))
ax2.plot(range(10, 0, -1));

plt.subplot(121)
plt.plot(range(10))
plt.subplot(122) 
plt.plot(range(10, 0, -1));

import numpy as np
x = np.linspace(-10, 10, 100)
y = x**2
plt.plot(x,y,'r--')  #Dashed red line; see table on p. 114.
plt.xlabel('$x$')  #LaTeX equations can be included by enclosing in $$.
plt.ylabel('$y$')
plt.title('A Parabola')
plt.legend(['$f(x)$'])  #Expects a list of strings.
plt.xlim(xmin=-8, xmax=8);  #Axis limits.
#plt.savefig('filename.svg')  #Save the plot to disk.

from matplotlib.patches import Polygon
import scipy.stats as stats  #The book likes to import it as `scs`.
a, b, c = -5, 5, stats.norm.ppf(0.05)
x = np.linspace(a, b, 100)
y = stats.norm.pdf(x)
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)
plt.plot(x, y, 'b', linewidth=2)
plt.ylim(ymin=0)
plt.xlim(xmin=a, xmax=b)
Ix = np.linspace(a, c)
Iy = stats.norm.pdf(Ix)
verts = [(a, 0)] + list(zip(Ix, Iy)) + [(c, 0)]
poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
ax.add_patch(poly)
ax.annotate('$p\%$', xy=(-2, 0.025), xytext=(-3, 0.1),
            arrowprops=dict(width=.5),
            )
plt.xlabel('$R_{t+1}$')
plt.ylabel('$f(R_{t+1})$')
ax.set_xticks([c, 0])
ax.set_xticklabels(['$-VaR_{t+1}^p$', '0'])
ax.set_yticks([])
plt.savefig('img/var.svg')
plt.close()

import pandas as pd
import pandas_datareader.data as web
import scipy.stats as stats  #The book likes to import it as `scs`.
p = web.DataReader("^GSPC", 'yahoo', start='1/1/2013', end='10/12/2017')['Adj Close']
r = np.log(p)-np.log(p).shift(1)
r.name = 'Return'
r = r[1:]  #Remove the first observation (NaN).
plt.figure(figsize=(12, 4))
plt.subplot(121)
sns.distplot(r, kde=False, fit=stats.norm)  #Histogram overlaid with a fitted normal density.
plt.subplot(122)
r.plot()  #Note that this is a pandas method! It looks prettier than plt.plot(r).
plt.savefig('img/stylizedfacts.svg')  #Save to file.
plt.close()

VaR_hist = -r.quantile(.01)  #Alternatively, VaR=np.percentile(r,1).
VaR_hist

ax = r.hist(bins=30)  #Another pandas method: histogram with 30 bins.
ax.set_xticks([-VaR_hist])
ax.set_xticklabels(['$-VaR_{t+1}^{0.01}$ = -%4.3f' %VaR_hist])  #4.3f means floating point with 4 digits, of which 3 decimals.
plt.title('Historical VaR')
plt.savefig('img/var_hist.svg')
plt.close()

mu, sig = stats.norm.fit(r)  #Fit a normal distribution to `r`.
VaR_norm = -mu-sig*stats.norm.ppf(0.01)
VaR_norm

ax = sns.distplot(r, kde=False, fit=stats.norm)  #Histogram overlaid with a fitted normal density.
ax.set_xticks([-VaR_norm])
ax.set_xticklabels(['$-VaR_{t+1}^{0.01}$ = -%4.3f' %VaR_norm])
ax.text(0.02,60,'$\mu=%7.6f$\n$\sigma=%7.6f$' %(mu, sig))  #\n means newline.
plt.title('Normal VaR')
plt.savefig('img/var_norm.svg')
plt.close()

x = np.linspace(-6, 6, 200)
df=[1, 2, 3, 10]
for nu in df:
    plt.plot(x, stats.t.pdf(x, nu))
legend = ['$\\nu=%1.0f$' % nu for nu in df]  #Double escaping: \\nu, not \nu, because \n is newline.
plt.plot(x, stats.norm.pdf(x))
legend.append('Normal')
plt.legend(legend)
plt.title("Student's $t$ Densities")
plt.xlabel('x')
plt.ylabel('$f_{\\nu}(x)$')
plt.savefig('img/tdists.svg')
plt.close()

df, m, h = stats.t.fit(r)  #Fit a location-scale t distribution to r.
VaR_t = -stats.t.ppf(0.01, df, loc=m, scale=h)
VaR_t

ax = sns.distplot(r, kde=False, fit=stats.t)  #Histogram overlaid with a fitted t density.
ax.set_xticks([-VaR_t])
ax.set_xticklabels(['$-VaR_{t+1}^{0.01}$ = -%4.3f' %VaR_t])
ax.text(0.02,60,'$m=%7.6f$\n$h=%7.6f$\n$\\nu=%7.6f$' %(m, h, df))
plt.title("Student's $t$ VaR")
plt.savefig('img/var_t.svg')
plt.close()

stats.chi2.ppf(0.95, 2)

stats.jarque_bera(r)  #Returns (JB, p-val).

#This is the manual way to do it.
x = np.linspace(.01,.99)
emp = r.quantile(x)
mu, sig = stats.norm.fit(r)
theo = stats.norm.ppf(x, mu, sig)
ax = plt.plot(theo, emp.values, 'o')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.title('QQ Plot vs. Normal')
plt.savefig('img/qq_norm.svg')
plt.close()

#this is a bit simpler.
sm.qqplot(r, dist=stats.t, fit=True)
plt.title("QQ Plot vs. Student's $t$")
plt.savefig('img/qq_t.svg')
plt.close()

sig2_hist = r.rolling(window=250).var()
sig2_hist.plot();

sig2_ewma = r.ewm(alpha=0.06).var()  #alpha=(1-lambda).
sig2_ewma.plot();

sig_ewma = np.sqrt(sig2_ewma)
mu = np.mean(r)
z = (r-mu)/sig_ewma
VaR_filtered_hist = -mu-sig_ewma*z.quantile(0.01)
VaR_filtered_hist.plot(color='red')
plt.plot(-r)
plt.title('Filtered Historical VaR');

import statsmodels.formula.api as smf
y = (r < -VaR_filtered_hist)*1  #Multiplication by 1 turns True/False into 1/0.
y.name='I'
data = pd.DataFrame(y)
model = smf.ols('I.subtract(0.01)~I.shift(1)', data=data)
res = model.fit()
print(res.summary2())

print(res.f_test('Intercept=0, I.shift(1)=0'))

