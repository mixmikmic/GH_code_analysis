import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

x_min = -15
x_max = 15
x = np.arange(x_min, x_max)
# x

y_min = -15
y_max = 15
y = np.arange(y_min, y_max)
# y

X, Y = np.meshgrid(x, y)

print("Shape of X: ", X.shape)
print("Shape of Y: ", Y.shape)

print()
# print("X: \n", X)
print()
# print("Y: \n", Y)

X = X.flatten()
Y = Y.flatten()

# print("X: \n", X)
# print("Y: \n", Y)

plt.figure(figsize=(11, 11))
plt.scatter(X, Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Cartesian 2D plane")

def create_circle_label(x, y, r):
    
    output = []
    
    for i, j in zip(x, y):
        
        if i**2+j**2 <= r**2:
            output.append(1)
        else:
            output.append(0)
            
    return output    

circle_label = create_circle_label(x=X, y=Y, r=8)
# circle_label

plt.figure(figsize=(11, 11))
plt.scatter(X, Y, c=circle_label)
plt.xlabel("x coordinates")
plt.ylabel("y coordinates")
plt.title("Circular area labels")



