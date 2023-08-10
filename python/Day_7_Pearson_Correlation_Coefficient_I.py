import math

def input_floats():
    return [float(i) for i in input().split(" ")]

def variance(x, m):
    return sum((i - m) ** 2 for i in x) / len(x)

def covariance(x, y, xm, ym):
    return sum((i - xm) * (j - ym) for i, j in zip(x, y)) / len(x)

#x = input_floats()
# y = input_floats()
x = [10, 9.8, 8, 7.8, 7.7, 7, 6, 5, 4, 2] 
y = [200, 44, 32, 24, 22, 17, 15, 12, 8, 4]
N = len(x)

x_mean = sum(x) / N
y_mean = sum(y)/ N

x_var = variance(x, x_mean)
y_var = variance(y, y_mean)

covxy = covariance(x, y, x_mean, y_mean)

pearson = covxy / (math.sqrt(x_var) * math.sqrt(y_var))

x_mean, x_var, math.sqrt(x_var)

y_mean, y_var, math.sqrt(y_var)

covxy

pearson

