import pandas as pd
import similarity as sm
a = pd.read_csv('sample data.csv')
a.head()

x,y = sm.createdict(a['Coverage'].tolist(),a['Education'].tolist())
sm.createdict(a['Coverage'].tolist(),a['Education'].tolist())

sm.euclids(x,y)

sm.manhattan(x,y)

sm.cosine(x,y)

sm.MahalanobisDist(x, y)

sm.compare(a,a)

