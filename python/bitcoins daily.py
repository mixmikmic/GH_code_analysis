from tpot import TPOTRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

filename='SPY Data.csv'
history_depth=3
header=1
delimiter=','
names=['Date','high','low','volume']
prediction_field='high'

bitcoins_daily=pd.read_csv(filename,delimiter=delimiter,header=header,names=names)
closes=bitcoins_daily[prediction_field].values.tolist()
features=np.array([closes[i:i+history_depth] for i in range(len(closes)) if i<(len(closes)-history_depth)])
labels=np.array([closes[i+history_depth] for i in range(len(closes)) if i<(len(closes)-history_depth)])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_daily_bitcoins.py')

