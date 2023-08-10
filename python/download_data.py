from sklearn.datasets import load_iris
from pyspark.mllib.classification import LabeledPoint
import numpy as np
data = load_iris()

data.keys()

x = data['data']
y = data['target']
y = y[:, np.newaxis]
xy = np.concatenate([y, x], 1)

x.shape

y.shape

xy.shape

x.tolist()

rdd = sc.parallelize(xy.tolist())

rdd.take(3)

ds = rdd.map(lambda pt: LabeledPoint(pt[0], pt[1:]))

ds.take(3)

np.savetxt("data.csv", xy, delimiter=",")

rows = []
with open("/home/hans/data.csv") as file:
    for line in file:
        rows.append(line.strip())

data = sc.parallelize(rows)

data.take(3)

ds = data.map(lambda row: LabeledPoint(int(row[0]), [float(x) for x in row[1:]]))

ds.take(3)

