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

# Read Train & Test Datasets
train = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('titanic-r/train.csv')
test = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('titanic-r/test.csv')

train.dtypes

train.describe().show()

train.show(2)

import pyspark.sql.functions as F
train_1 = train.select(train['PassengerId'], 
                 train['Survived'].cast("integer").alias("Survived"),
                 train['Pclass'].cast("integer").alias("Pclass"),
                 F.when(train['Sex'] == 'female', 1).otherwise(0).alias("Gender"), 
                 train['Age'].cast("integer").alias("Age"),
                 train['SibSp'].cast("integer").alias("SibSp"),
                 train['Parch'].cast("integer").alias("Parch"),
                 train['Fare'].cast("float").alias("Fare"))

train.count()

train_1.count()

train_1.show(2)

train_1.describe().show()

# Replace null age by 30
# Do we have nulls ?
train_1.filter(train_1['Age'].isNull()).show(40)

# Replace null age by 30
train_1.na.fill(30,'Age').show(40)

# Replace null age by 30
train_2 = train_1.na.fill(30,'Age')

train_2.crosstab("Gender","Survived").show()

print "F = %3.2f%% M = %3.2f%%" % ( (100*233.0/(233+81)), (100*109.0/(109+468)) )

#
# 1 : Simple Model (M=Survived) 
#
test.show(2)

out = test.select(test['PassengerId'], 
                 F.when(test['Sex'] == 'female', 1).otherwise(0).alias("Survived"))

out.show(2)

out.coalesce(1).write.mode('overwrite').format('com.databricks.spark.csv').options(header='true').save('titanic-r/spark-sub-01.csv')

# Submit
# Rank : 2586 Score : 0.76555

#
# Would age be a better predictor ?
#
train_1.na.drop().crosstab("Age","Survived").show()

#
# *** Home work : See if Pclass, SibSp or Parch is a better indication and change survival accordingly¶
#

from pyspark.mllib.regression import LabeledPoint
def parse_passenger_list(r):
    return LabeledPoint(r[1],[r[2],r[3],r[4],r[5],r[6],r[7]])

train_rdd = train_2.map(lambda x: parse_passenger_list(x))

train_rdd.count()

train_rdd.first()

from pyspark.mllib.tree import DecisionTree
model = DecisionTree.trainClassifier(train_rdd, numClasses=2,categoricalFeaturesInfo={})

print(model)
# print(model.toDebugString())

# Transform test and predict
import pyspark.sql.functions as F
test_1 = test.select(test['PassengerId'], 
                 test['Pclass'].cast("integer").alias("Pclass"),
                 F.when(test['Sex'] == 'female', 1).otherwise(0).alias("Gender"), 
                 test['Age'].cast("integer").alias("Age"),
                 test['SibSp'].cast("integer").alias("SibSp"),
                 test['Parch'].cast("integer").alias("Parch"),
                 test['Fare'].cast("float").alias("Fare"))

test_1.show(2)

# Do we have nulls ?
test_1.filter(test_1['Age'].isNull()).show(40)

test_1.groupBy().avg('Age').show()

# Replace null age by 30.24 - the mean
test_2 = test_1.na.fill(30,'Age')

# parse test data for predictions
from pyspark.mllib.regression import LabeledPoint
def parse_test(r):
    return (r[1],r[2],r[3],r[4],r[5],r[6])

test_rdd = test_2.map(lambda x: parse_test(x))

test_rdd.count()

predictions = model.predict(test_rdd)

predictions.first()

out_rdd = test_2.map(lambda x: x[0]).zip(predictions)

out_rdd.first()

out_df = out_rdd.toDF(['PassengerId','Survived'])

out_df.show(2)

out_1 = out_df.select(out_df['PassengerId'],
                      out_df['Survived'].cast('integer').alias('Survived'))

out_1.show(2)

out_1.coalesce(1).write.mode('overwrite').format('com.databricks.spark.csv').options(header='true').save('titanic-r/spark-sub-02.csv')

# Submit
# Rank : 2038 +549 Score : 0.77512

from pyspark.mllib.tree import RandomForest
model_rf = RandomForest.trainClassifier(train_rdd, numClasses=2,categoricalFeaturesInfo={},numTrees=42)

print(model_rf)
#print(model_rf.toDebugString())

pred_rf = model_rf.predict(test_rdd).coalesce(1)

pred_rf.first()

out_rf = test_2.map(lambda x: x[0]).coalesce(1).zip(pred_rf)

out_rf.first()

out_df_rf = out_rf.toDF(['PassengerId','Survived'])

out_2 = out_df_rf.select(out_df_rf['PassengerId'],
                      out_df_rf['Survived'].cast('integer').alias('Survived'))

out_2.coalesce(1).write.mode('overwrite').format('com.databricks.spark.csv').options(header='true').save('titanic-r/spark-sub-03.csv')

# Submit
# Rank : 1550 +488 Score : 0.78469

# Looks like we are on a roll ! Let us try SVM !

from pyspark.mllib.classification import SVMWithSGD
model_svm = SVMWithSGD.train(train_rdd, iterations=100)

pred_svm = model_svm.predict(test_rdd).coalesce(1)
out_svm = test_2.map(lambda x: x[0]).coalesce(1).zip(pred_svm)
out_df_svm = out_svm.toDF(['PassengerId','Survived'])

out_3 = out_df_svm.select(out_df_svm['PassengerId'],
                      out_df_svm['Survived'].cast('integer').alias('Survived'))

out_3.coalesce(1).write.mode('overwrite').format('com.databricks.spark.csv').options(header='true').save('titanic-r/spark-sub-04.csv')

# Not good. Only 0.39713 !



