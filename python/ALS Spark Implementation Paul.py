import os, sys, numpy as np
os.environ['SPARK_HOME']="/Users/paulthompson/spark-1.6.1-bin-hadoop2.4"
sys.path.append("/Users/paulthompson/spark-1.6.1-bin-hadoop2.4/python/")
from pyspark import SparkConf, SparkContext
conf = (SparkConf().setMaster("local").setAppName("My app").set("spark.executor.memory", "1g"))
from pyspark.sql import SQLContext
from pyspark.sql.dataframe import StructType, StructField, IntegerType, FloatType
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

mv_ratings = sc.textFile('/Users/paulthompson/Documents/MSAN_Files/Spr2_Distributed/HW1/movies/ratings.txt')

# Creating sparse representation of A matrix with users as rows and items as columns
user_item_ratings = mv_ratings.map(lambda line: (int(line.split(':')[0]), (int(line.split(':')[1]), line.split(':')[2])))
user_item_ratings = user_item_ratings.groupByKey()
print user_item_ratings.take(1)

# Creating sparse representation of transposed A matrix with items as rows and rows as columns
item_user_ratings = mv_ratings.map(lambda line: (int(line.split(':')[1]), (int(line.split(':')[0]), line.split(':')[2])))
item_user_ratings = item_user_ratings.groupByKey()
print item_user_ratings.take(1)

print "Number of Items (columns of A matrix):", item_user_ratings.count()
print "Number of Users (rows of A matrix):", user_item_ratings.count()

# User Defined Parameters
lambda_ = sc.broadcast(0.1) # Regularization parameter
n_factors = sc.broadcast(3) # nfactors of User matrix and Item matrix
n_iterations = 20 # How many times to iterate over the user and item matrix calculations.

# Initizializing Items Matrix (User matrix doesn't need to be initialized since it is solved for first):
Items = item_user_ratings.map(lambda line: (line[0], 5 * np.random.rand(1, n_factors.value)))

print Items.take(10)

# The item matrix is needed in all partitions when solving for rows of User matrix individually
Items_broadcast = sc.broadcast({
  k: v for (k, v) in Items.collect()
})

j = 0
for k, v in {k: v for (k, v) in Items.collect()}.iteritems():
    print k, v
    j+=1
    if j > 10:
        break

j = 0
for i in user_item_ratings.take(1)[0][1]:
    print i
    j+=1
    if j > 10:
        break

def Update_User(userTuple):
    '''
    This function calculates (userID, Users[i]) using:
        'Users[i] = inverse(Items*Items^T + I*lambda) * Items * A[i]^T'
    Dot product calculations are done differently than normal to allow for sparsity. Rather 
    than row of left matrix times column of right matrix, sum result of column of left matrix  
    * rows of right matrix (skipping items for which user doesn't have a rating).
    '''
    Itemssquare = np.zeros([n_factors.value,n_factors.value])
    for matrixA_item_Tuple in userTuple[1]:
        itemRow = Items_broadcast.value[matrixA_item_Tuple[0]][0]
        for i in range(n_factors.value):
            for j in range(n_factors.value):
                Itemssquare[i,j] += float(itemRow[i]) * float(itemRow[j])
    leftMatrix = np.linalg.inv(Itemssquare + lambda_.value * np.eye(n_factors.value))
    rightMatrix = np.zeros([1,n_factors.value])
    for matrixA_item_Tuple in userTuple[1]:
        for i in range(n_factors.value):
            rightMatrix[0][i] += Items_broadcast.value[matrixA_item_Tuple[0]][0][i] * float(matrixA_item_Tuple[1])
    newUserRow = np.dot(leftMatrix, rightMatrix.T).T
    return (userTuple[0], newUserRow)

Users = user_item_ratings.map(Update_User)

print Users.take(1)

# The item matrix is needed in all partitions when solving for rows of User matrix individually
Users_broadcast = sc.broadcast({
  k: v for (k, v) in Users.collect()
})

def Update_Item(itemTuple):
    '''
    This function calculates (userID, Users[i]) using:
        'Users[i] = inverse(Items*Items^T + I*lambda) * Items * A[i]^T'
    Dot product calculations are done differently than normal to allow for sparsity. Rather 
    than row of left matrix times column of right matrix, sum result of column of left matrix  
    * rows of right matrix (skipping items for which user doesn't have a rating).
    '''
    Userssquare = np.zeros([n_factors.value,n_factors.value])
    for matrixA_user_Tuple in itemTuple[1]:
        userRow = Users_broadcast.value[matrixA_user_Tuple[0]][0]
        for i in range(n_factors.value):
            for j in range(n_factors.value):
                Userssquare[i,j] += float(userRow[i]) * float(userRow[j])
    leftMatrix = np.linalg.inv(Userssquare + lambda_.value * np.eye(n_factors.value))
    rightMatrix = np.zeros([1,n_factors.value])
    for matrixA_user_Tuple in itemTuple[1]:
        for i in range(n_factors.value):
            rightMatrix[0][i] += Users_broadcast.value[matrixA_user_Tuple[0]][0][i] * float(matrixA_user_Tuple[1])
    newItemRow = np.dot(leftMatrix, rightMatrix.T).T
    return (itemTuple[0], newItemRow)

Items = item_user_ratings.map(Update_Item)

print Items.take(1)

Items_broadcast = sc.broadcast({
  k: v for (k, v) in Items.collect()
})

def getRowSumSquares(userTuple):
    userRow = Users_broadcast.value[userTuple[0]]
    rowSSE = 0.0
    for matrixA_item_Tuple in userTuple[1]:
        predictedRating = 0.0
        for i in range(n_factors.value):
            predictedRating += userRow[0][i] * Items_broadcast.value[matrixA_item_Tuple[0]][0][i]
        SE = (float(matrixA_item_Tuple[1]) - predictedRating) ** 2
        rowSSE += SE
    return rowSSE

        

SSE = user_item_ratings.map(getRowSumSquares).reduce(lambda a, b: a + b)
Count = mv_ratings.count()
MSE = SSE / Count
print "MSE:", MSE

for iter in range(n_iterations):
    Users = user_item_ratings.map(Update_User)
    Users_broadcast = sc.broadcast({k: v for (k, v) in Users.collect()})
    Items = item_user_ratings.map(Update_Item)
    Items_broadcast = sc.broadcast({k: v for (k, v) in Items.collect()})
    SSE = user_item_ratings.map(getRowSumSquares).reduce(lambda a, b: a + b)
    MSE = SSE / Count
    print "MSE:", MSE



