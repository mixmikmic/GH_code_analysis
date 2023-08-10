# prepare data
import numpy as np

# load matrix A from file
filepath = '../data/letters.txt'
matrixA = np.loadtxt(filepath).T
print('The matrix A has ', matrixA.shape[0], ' rows and ', matrixA.shape[1], ' columns.')

def countSketchInMemroy(matrixA, s):
    m, n = matrixA.shape
    matrixC = np.zeros([m, s])
    hashedIndices = np.random.choice(s, n, replace=True)
    randSigns = np.random.choice(2, n, replace=True) * 2 - 1 # a n-by-1{+1, -1} vector
    matrixA = matrixA * randSigns.reshape(1, n) # flip the signs of 50% columns of A
    for i in range(s):
        idx = (hashedIndices == i)
        matrixC[:, i] = np.sum(matrixA[:, idx], 1)
    return matrixC

s = 50 # sketch size, can be tuned
matrixC = countSketchInMemroy(matrixA, 50)

# Test
# compare the l2 norm of each row of A and C
rowNormsA = np.sqrt(np.sum(np.square(matrixA), 1))
print(rowNormsA)
rowNormsC = np.sqrt(np.sum(np.square(matrixC), 1))
print(rowNormsC)

def countSketchStreaming(matrixA, s):
    m, n = matrixA.shape
    matrixC = np.zeros([m, s])
    hashedIndices = np.random.choice(s, n, replace=True)
    randSigns = np.random.choice(2, n, replace=True) * 2 - 1 
    for j in range(n):
        a = matrixA[:, j]
        h = hashedIndices[j]
        g = randSigns[j]
        matrixC[:, h] += g * a
    return matrixC

s = 50 # sketch size, can be tuned
matrixC = countSketchStreaming(matrixA, s)

# Test
# compare the l2 norm of each row of A and C
rowNormsA = np.sqrt(np.sum(np.square(matrixA), 1))
print(rowNormsA)
rowNormsC = np.sqrt(np.sum(np.square(matrixC), 1))
print(rowNormsC)

def countSketchMapReduce(filepath, s):
    # load data
    rawRDD = sc.textFile(filepath)
    # parse string data to vectors
    vectorRDD = rawRDD.map(lambda l: np.asfarray(l.split()))
    # map the vectors to key-value pairs
    pairRDD = vectorRDD.map(lambda vect: (np.random.randint(0, s), (np.random.randint(0, 2) * 2 - 1) * vect ))
    # reducer
    vectList = pairRDD.reduceByKey(lambda v1, v2: v1+v2).map(lambda pair: pair[1]).collect()
    return np.asarray(vectList).T

s = 50 # sketch size, can be tuned
filepath = './data/letters.txt'
matrixC = countSketchMapReduce(filepath, s)



