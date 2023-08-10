from matplotlib import pyplot as plt
import pandas as pd

data = [1, 5, 2, 3, 2]
df = pd.DataFrame(data, columns=['value'])

df

plt.bar(df.index, df.value)

# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.bar.html
plt.bar(df.index, df.value, capstyle='round')

plt.bar(df.index, df.value, joinstyle='round')

