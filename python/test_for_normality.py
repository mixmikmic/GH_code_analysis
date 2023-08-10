get_ipython().run_line_magic('matplotlib', 'inline')
import pyodbc
import pandas as pd
import xlwings as xw
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

wb = xw.Workbook.active()
data = xw.Range('A1').table.value
df = pd.DataFrame(data=data[1:], columns=data[0])
df.sort_values(by=['VIN','DAYS_TO_FAIL_MINZERO'], inplace=True)
df.head()

df.MILES_TO_FAIL.plot.hist()
plt.show()

df.Array1.describe()

z = (df.MILES_TO_FAIL-df.MILES_TO_FAIL.mean())/df.MILES_TO_FAIL.std()
stats.probplot(z, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()

help(stats.normaltest)

statistic, pvalue = stats.normaltest(df.Array1)
if pvalue > 0.05:
    print("Data most likely is normally distributed")
else:
    print("Data is not likely to be normally distributed")

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6,8))

ax1 = plt.subplot(211)
df.Array1.plot.hist()
ax1.set_title("Histogram")

ax2 = plt.subplot(212)
z = (df.Array1-df.Array1.mean())/df.Array1.std()
stats.probplot(z, dist="norm", plot=plt,)
ax2.set_title("Normal Q-Q Plot")

plt.show()

