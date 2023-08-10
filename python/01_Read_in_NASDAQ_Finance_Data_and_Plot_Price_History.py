import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame()
df = df.from_csv('stock_data/tsla.csv')

df.head()

df.dtypes

plt.figure(figsize=(10, 4))
df['close'].plot()
plt.show()

