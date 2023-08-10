from sklearn import linear_model
import numpy as np

x = [[5, 7], [6, 6], [7, 4], [8, 5], [9, 6]]
y = [10, 20, 60, 40, 50]

# Create linear regression object
lm = linear_model.LinearRegression()

# Train the model using the training sets
lm.fit(x, y)
a = lm.intercept_
b = lm.coef_
print(a, b[0], b[1])

'''
Sample Input

2 7
0.18 0.89 109.85
1.0 0.26 155.72
0.92 0.11 137.66
0.07 0.37 76.17
0.85 0.16 139.75
0.99 0.41 162.6
0.87 0.47 151.77
4
0.49 0.18
0.57 0.83
0.56 0.64
0.76 0.18

Sample Output

105.22
142.68
132.94
129.71
'''
print('---Input value---')
m, n = [int(i) for i in input().split()]
X = []
Y = []
data = []
for i in range(n):
    data.append(([float(i) for i in input().split()]))

import numpy as np
A = np.array(data)
X = A[:, :-1]
Y = A[:, -1]

from sklearn import linear_model

lm = linear_model.LinearRegression()
lm.fit(X, Y)

# q denotes the number of feature sets (f1, f2)
q = int(input())

# value of feature sets
f = []
for i in range(q):
    f.append([float(i) for i in input().split()])

# print out the predicted value of Y

print('---Predicted results---')
for y in lm.predict(f):
    print(round(y, 2))

