get_ipython().magic('load_ext watermark')

get_ipython().magic('watermark -a "Michal Dyzma" -t -d -v -m -p numpy,scipy,pandas')

# Basic pyton packages import
import pandas as pd
import numpy as np
import sympy as smp

# graphs appear in the notebook
get_ipython().magic('matplotlib inline')
# symbolic math as LaTeX symbols
smp.init_printing(use_unicode=True)

# generate data
# absorbance of the sample at 50 nm intervals between 350-700 nm.
x = range(350, 710, 10)

#random values 
y = np.random.uniform(0,0.7,len(x))

print(y)
print(list(x))

#I stored batch of data n csv file to mimic wet lab procedure

#Read measurements from file
df = pd.read_csv("data/lambda_max.csv", header=None, names=["Absorbance"])
df.describe()

# Find maximal wavelenght for which absorbance is highest
df['Absorbance'].argmax()

ax = df.Absorbance.plot(title="$\lambda_{max}$ of the protein")
ax.set_ylabel("Absorbance")
ax.set_xlabel("wavelength [nm]")

# Abs_410 = 0.683876
epsilon = 0.683876/(0.110*1)
epsilon

# Calibration curve
concentration = np.array([5, 10, 25, 30, 40, 50, 60, 70])
absorbance = np.array([0.106, 0.236, 0.544, 0.690, 0.791, 0.861, 0.882, 0.911])

import matplotlib.pyplot as plt
plt.plot(concentration, absorbance, 'o')
plt.grid(alpha=0.8)

# Linear regression
from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(concentration, absorbance)
print("r-squared: {}".format(r_value**2))
print(slope)
print(intercept)

x = (absorbance - intercept)/slope

plt.plot(concentration, absorbance, 'o', label='data points')
plt.plot(concentration, intercept + slope*concentration, 'r', label='fitted line')
plt.legend()
plt.grid(alpha=0.8)

x, y = smp.symbols('x y')
A, b = smp.symbols('A b', float=True)

expr = (A*x+b)-y

smp.solveset(expr, x)

#polynomial fit
import numpy.polynomial.polynomial as P

coefs2 = P.polyfit(concentration, absorbance, 2)
ffit2 = P.polyval(concentration, coefs2)

plt.plot(concentration, absorbance, 'o', label='data points')
plt.plot(concentration, ffit2, 'r', label='2nd poly')

plt.legend()
plt.grid(alpha=0.8)

print(coefs2)

coefs2 = P.polyfit(concentration, absorbance, 2)
ffit2 = P.polyval(concentration, coefs2)

coefs3 = P.polyfit(concentration, absorbance, 3)
ffit3 = P.polyval(concentration, coefs3)

plt.plot(concentration, absorbance, 'o', label='data points')
plt.plot(concentration, ffit2, 'g', label='2nd poly')
plt.plot(concentration, ffit3, 'r', label='3rd poly')

plt.legend()
plt.grid(alpha=0.8)

x, y = smp.symbols('x y')
a1, a2, a3 = smp.symbols('a1 a2 a3', float=True)

expr = (a1*x**2+a2*x+a3)-y
smp.solveset(expr, x)

from numpy.polynomial import Polynomial as P

p = P.fit(concentration, absorbance, 2)
(p - .5).roots()

