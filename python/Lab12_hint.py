get_ipython().magic('matplotlib inline')
import numpy as np

P = np.matrix(
[[ 0.1  , 0.2 , 0.7],
 [ 0.01 ,  0. , 0.99 ],
 [ 0.4  ,  0.4,  0.2 ]
])

## There is a difference between the following two pieces of the code. 
## Make sure you use the right one


## Code number 1
A2 = np.copy(P)
for i in np.arange(1,100):
    A1 = A2
    A2 = A1*P
    E1 = np.round(A1,5)
    E2 = np.round(A2,5)
    if(np.all(E1 == E2)):
        break

print(E1)
print(E2)
print(E1 == E2)
print(A1)
print(A2)
print(A1 == A2)
print(i)

## Code number 2
A2 = np.copy(P)
for i in np.arange(1,100):
    A1 = A2
    A2 = np.round(A1*P,5)
    
    if(np.all(A1 == A2)):
        break



print(A1)
print(A2)
print(A1 == A2)
print(i)

