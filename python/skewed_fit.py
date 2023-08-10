get_ipython().magic('pylab --no-import-all inline')

from scipy.special import erf
from scipy.optimize import curve_fit

matplotlib.rcParams["figure.figsize"] = (10, 6)
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams["font.family"] = "sans"
matplotlib.rcParams["font.size"] = 20

def normpdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def normcdf(x, mu, sigma, alpha=1):
    return 0.5 * (1 + erf(alpha * (x - mu) / (sigma * np.sqrt(2))))

def skewed(x, mu, sigma, alpha, a):
    return a * normpdf(x, mu, sigma) * normcdf(x, mu, sigma, alpha)

get_ipython().magic('pinfo ax1.set_ylabel')

fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(14, 6))
x = np.linspace(-5, 5, 200)
for alpha in [0, 1, 2, 5, 10]:
    ax1.plot(x, skewed(x, mu=1, sigma=1, alpha=alpha, a=1), label=r"$\alpha$=%d" % alpha)
ax1.set_title("Positive skewness")
ax1.legend(loc="upper left", fontsize=16)
for alpha in [0, 1, 2, 5, 10]:
    ax2.plot(x, skewed(x, mu=1, sigma=1, alpha=-alpha, a=1), label=r"$\alpha$=%d" % -alpha)
ax2.legend(loc="upper left", fontsize=16)
ax2.set_title("Negative skewness")
ax2.set_yticklabels([])
ax2.set_xlim(-4, 6)
fig.subplots_adjust(wspace=0)

help(curve_fit)

# produce data to fit
sigma = 10
mu = 50
alpha = 2
a = 1
x = np.linspace(0,100,100)

y = skewed(x, mu, sigma, alpha=alpha, a=a)
yn = y + 0.002 * np.random.normal(size=len(x))

# fit to skewed distribution
sopt, scov = curve_fit(skewed, x, yn, p0=(20, 20, 1, 1))
y_fit= skewed(x, *sopt)

# fit to normal distribution
gopt, gcov = curve_fit(normpdf, x, yn, p0=(20, 20))
y_gfit = normpdf(x, *gopt)

# plot
#plt.plot(x, y, "r-")
plt.plot(x, yn, "bo", label="data")
plt.plot(x, y_fit, "g", label="fit skewed")
plt.plot(x, y_gfit, "r", label="fit normal")
plt.legend(loc="upper left", fontsize=16)

# parameters
print("Parameters skewed :")
print("-------------------")
print("mu    :", sopt[0])
print("sigma :", sopt[1])
print("alpha :", sopt[2])
print("a     :", sopt[3])
print("\nParameters normal :")
print("-------------------")
print("mu    :", gopt[0])
print("sigma :", gopt[1])

