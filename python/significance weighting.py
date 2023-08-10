import csv
import pandas as pd

data = '../data/web.train2.0.csv'
data1 = '../data/movie.train2.0.csv'

webtrain = pd.read_csv(data, header= 0)
movietrain = pd.read_csv(data1, header = 0)

m = webtrain.shape[0]
a = []
print(m)

##### webtrain
for x in range(0, m):
    for y in range(0,m):
        a.append(pd.Series.count(list(set(webtrain.iloc[x]).intersection(set(webtrain.iloc[y])))))

movietrain_numpyarray = pd.DataFrame.as_matrix(movietrain)

m1 = movietrain.shape[0]
n1 = movietrain.shape[1]

### movie
a1 = []
for x in range(0,m1):
    for y in range(0,x):
        a1.append(sum(movietrain_numpyarray[x,] == movietrain_numpyarray[y,]))
        print(x,y)

outfile1 = open('../output/movie_train_a.csv','w')
out = csv.writer(outfile1)
out.writerows(map(lambda x: [x], a1))
outfile1.close()

outfile = open('../output/webtrain_a.csv','w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], a))
outfile.close()

