import datetime
from pytz import timezone
print "Last run @%s" % (datetime.datetime.now(timezone('US/Pacific')))
#
from pyspark.context import SparkContext
print "Running Spark Version %s" % (sc.version)
#
from pyspark.conf import SparkConf
conf = SparkConf()
print conf.toDebugString()

# Read Dataset
freq_df = sqlContext.read.format('com.databricks.spark.csv')            .options(header='true')            .load('freq-flyer/AirlinesCluster.csv')

freq_df.show(5)

freq_df.count()

freq_df.dtypes

from numpy import array
freq_rdd = freq_df.map(lambda row: array([float(x) for x in row]))

freq_rdd.take(3)

from pyspark.mllib.clustering import KMeans
from math import sqrt

freq_rdd.first()
# Balance, TopStatusQualMiles, NonFlightMiles, NonFlightTrans, FlightMiles, FlightTrans, DaysSinceEnroll

help(KMeans.train)

km_mdl_1 = KMeans.train(freq_rdd, 2, maxIterations=10,runs=10, initializationMode="random")

for x in km_mdl_1.clusterCenters:
        print "%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (x[0],x[1],x[2],x[3],x[4],x[5],x[6])
# Balance, TopStatusQualMiles, NonFlightMiles, NonFlightTrans, FlightMiles, FlightTrans, DaysSinceEnroll

for x in freq_rdd.take(10):
    print x,km_mdl_1.predict(x)

def squared_error(mdl, point):
    center = mdl.centers[mdl.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = freq_rdd.map(lambda point: squared_error(km_mdl_1,point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

from pyspark.mllib.stat import Statistics
summary = Statistics.colStats(freq_rdd)
print summary.mean()

print "Mean : %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (summary.mean()[0],summary.mean()[1],summary.mean()[2],
                                                            summary.mean()[3],summary.mean()[4],summary.mean()[5],
                                                            summary.mean()[6])
print "Max  : %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (summary.max()[0],summary.max()[1],
                                                                       summary.max()[2],
                                                            summary.max()[3],summary.max()[4],summary.max()[5],
                                                            summary.max()[6])
print "Min  : %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (summary.min()[0],summary.min()[1],
                                                                       summary.min()[2],
                                                            summary.min()[3],summary.min()[4],summary.min()[5],
                                                            summary.min()[6])
print "Variance : %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (summary.variance()[0],summary.variance()[1],
                                                                       summary.variance()[2],
                                                            summary.variance()[3],summary.variance()[4],summary.variance()[5],
                                                            summary.variance()[6])
# Balance, TopStatusQualMiles, NonFlightMiles, NonFlightTrans, FlightMiles, FlightTrans, DaysSinceEnroll

# You see, K-means clustering is "isotropic" in all directions of space and therefore tends to produce 
# more or less round (rather than elongated) clusters. [Ref 2]
# In this situation leaving variances unequal is equivalent to putting more weight on variables with smaller variance, 
# so clusters will tend to be separated along variables with greater variance. [Ref 3]
#
# center, scale, box-cox, preprocess in caret
# zero mean and unit variance
#
# (x - mu)/sigma
# org.apache.spark.mlib.feature.StandardScaler does this, but to the best of my knowledge 
#            as of now (9/28/14) not available for python 
# So we do it manually, gives us a chance to do some functional programming !
#

data_mean = summary.mean()
data_sigma = summary.variance()

for x in data_sigma:
    print x,sqrt(x)

def center_and_scale(a_record):
    for i in range(len(a_record)):
        a_record[i] = (a_record[i] - data_mean[i])/sqrt(data_sigma[i]) # (x-mean)/sd
    return a_record

freq_norm_rdd = freq_rdd.map(lambda x: center_and_scale(x))

freq_norm_rdd.first()

# now let us try with the standardized data
km_mdl_std = KMeans.train(freq_norm_rdd, 2, maxIterations=10,runs=10, initializationMode="random")

for x in km_mdl_std.clusterCenters:
        print "%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (x[0],x[1],x[2],x[3],x[4],x[5],x[6])
# Balance, TopStatusQualMiles, NonFlightMiles, NonFlightTrans, FlightMiles, FlightTrans, DaysSinceEnroll

WSSSE = freq_norm_rdd.map(lambda point: squared_error(km_mdl_std,point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# Let us try with k= 5 clusters instead of k=2
km_mdl_std_5 = KMeans.train(freq_norm_rdd, 5, maxIterations=10,runs=10, initializationMode="random")

for x in km_mdl_std_5.clusterCenters:
        print "%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (x[0],x[1],x[2],x[3],x[4],x[5],x[6])
# Balance, TopStatusQualMiles, NonFlightMiles, NonFlightTrans, FlightMiles, FlightTrans, DaysSinceEnroll

WSSSE = freq_norm_rdd.map(lambda point: squared_error(km_mdl_std_5,point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

km_mdl_std_10 = KMeans.train(freq_norm_rdd, 10, maxIterations=10,runs=10, initializationMode="random")
for x in km_mdl_std_10.clusterCenters:
        print "%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (x[0],x[1],x[2],x[3],x[4],x[5],x[6])
#
WSSSE = freq_norm_rdd.map(lambda point: squared_error(km_mdl_std_10,point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

cluster_rdd = freq_norm_rdd.map(lambda x: km_mdl_std_5.predict(x))

cluster_rdd.take(10)

freq_rdd_1 = inp_file.map(lambda line: array([int(x) for x in line.split(',')]))
freq_cluster_map = freq_rdd_1.zip(cluster_rdd)
freq_cluster_map.take(5) 
# Gives org.apache.spark.SparkException: Can only zip RDDs with same number of elements in each partition

freq_cluster_map = inp_file.map(lambda line: array([int(x) for x in line.split(',')])).zip(cluster_rdd)
freq_cluster_map.take(5) 
# Gives org.apache.spark.SparkException: Can only zip RDDs with same number of elements in each partition

freq_cluster_map = freq_rdd.zip(cluster_rdd)
freq_cluster_map.take(5) # This works !

cluster_0 = freq_cluster_map.filter(lambda x: x[1] == 0)
cluster_1 = freq_cluster_map.filter(lambda x: x[1] == 1)
cluster_2 = freq_cluster_map.filter(lambda x: x[1] == 2)
cluster_3 = freq_cluster_map.filter(lambda x: x[1] == 3)
cluster_4 = freq_cluster_map.filter(lambda x: x[1] == 4)

print cluster_0.count()
print cluster_1.count()
print cluster_2.count()
print cluster_3.count()
print cluster_4.count()

cluster_0.count()+cluster_1.count()+cluster_2.count()+cluster_3.count()+cluster_4.count()

freq_rdd_1.count()

freq_cluster_map.count()

cluster_0.take(5)

cluster_1.take(5)

cluster_2.take(5)

cluster_3.take(5)

cluster_4.take(5)

stat_0 = Statistics.colStats(cluster_0.map(lambda x: x[0]))
stat_1 = Statistics.colStats(cluster_1.map(lambda x: x[0]))
stat_2 = Statistics.colStats(cluster_2.map(lambda x: x[0]))
stat_3 = Statistics.colStats(cluster_3.map(lambda x: x[0]))
stat_4 = Statistics.colStats(cluster_4.map(lambda x: x[0]))
print "0 : %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (stat_0.mean()[0],stat_0.mean()[1],stat_0.mean()[2],
                                                            stat_0.mean()[3],stat_0.mean()[4],stat_0.mean()[5],
                                                            stat_0.mean()[6])
print "1 : %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (stat_1.mean()[0],stat_1.mean()[1],stat_1.mean()[2],
                                                            stat_1.mean()[3],stat_1.mean()[4],stat_1.mean()[5],
                                                            stat_1.mean()[6])
print "2 : %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (stat_2.mean()[0],stat_2.mean()[1],stat_2.mean()[2],
                                                            stat_2.mean()[3],stat_2.mean()[4],stat_2.mean()[5],
                                                            stat_2.mean()[6])
print "3 : %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (stat_3.mean()[0],stat_3.mean()[1],stat_3.mean()[2],
                                                            stat_3.mean()[3],stat_3.mean()[4],stat_3.mean()[5],
                                                            stat_3.mean()[6])
print "4 : %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (stat_4.mean()[0],stat_4.mean()[1],stat_4.mean()[2],
                                                            stat_4.mean()[3],stat_4.mean()[4],stat_4.mean()[5],
                                                            stat_4.mean()[6])
# Balance, TopStatusQualMiles, NonFlightMiles, NonFlightTrans, FlightMiles, FlightTrans, DaysSinceEnroll

# Different runs will produce different clusters
# Once the model is executed, the characteristics can interpreted & used in business





