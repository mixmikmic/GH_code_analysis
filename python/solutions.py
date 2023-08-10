import pandas as pd
from numpy import log, median
df = pd.DataFrame({'a': [1,2,3,4,5]})
df['log_a'] = df.a.apply(log)
assert log(df.a.median()) == df.log_a.median()    # this raises an error if false
print(log(df.a.median()), df.log_a.median())
print(log(df.a.std()), df.log_a.std())

EPS = 0.00001
df2 = pd.DataFrame({'b': [0,1,2,3,4,5]})
df2['log_b'] = df2.b.apply(lambda k: log(k + EPS) if k == 0 else log(k))
assert log(df2.b.median()) == df2.log_b.median()    # error!

print(log(df2.b.median()), df2.log_b.median())



