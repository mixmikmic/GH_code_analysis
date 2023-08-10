import pyspark
import matplotlib.pyplot as plt

from pyspark import SparkContext, SQLContext
sc = pyspark.SparkContext()
sqlc = SQLContext(sc)

from pyspark.sql.functions import split, explode
from pyspark.sql.functions import UserDefinedFunction as udf
from pyspark.sql.types import StringType,FloatType, StructType, StructField, IntegerType
from pyspark.sql import functions as F, Window
from pyspark.sql.functions import *

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd

predictions_train=sqlc.read.json('./data/ur_forks_stars_rec.json')

predictions_train.count()

test_set=sqlc.read.csv('./data/forks_stars_sample_test.csv')

test_set = test_set.withColumnRenamed("_c0", "user_id")
test_set = test_set.withColumnRenamed("_c1", "repo_id")
test_set = test_set.withColumnRenamed("_c2", "created_at")
test_set= test_set.withColumnRenamed("_c3","event")

test_set.show()

predictions_train.show()

predictions_train=predictions_train.withColumn('scores',explode('rec'))
predictions_train=predictions_train.withColumn('repo_id',predictions_train.scores['item'])                       .withColumn('score',predictions_train.scores['score'])

predictions_train=predictions_train.select('user_id','repo_id','score')

w=(Window.partitionBy('user_id').orderBy(col('score').desc()).rowsBetween(Window.unboundedPreceding, Window.currentRow))
predictions_train=predictions_train.withColumn('reccomendations',F.count('user_id').over(w))

predictions_train.show()

predictions_final=predictions_train.filter('score>0')

predictions_final=predictions_final.select(col('user_id').alias('p_user_id'),col('repo_id').alias('p_repo_id'),'score')

test_forks=test_set.filter(test_set.event=='fork')

test_join=test_forks.join(predictions_final,(test_forks.user_id==predictions_final.p_user_id)                   & (test_forks.repo_id==predictions_final.p_repo_id),'left')

test_join.filter('p_repo_id is not null').count()/test_forks.count()  

test_scores=test_join.filter('p_repo_id is not null')

l=test_scores.select('score').rdd.map(lambda x: x.score).collect()

l_norm=(l-np.min(l))/(np.max(l)-np.min(l))

pd.DataFrame(l_norm,columns=['score']).plot.hist() #1 user has 14.000 stars
#fig.title('Distribution of stars by user')
plt.xlabel('scores', fontsize=14)
plt.ylabel('number of tests', fontsize=14)
plt.title('UR scores test', fontsize=17)

plt.savefig('ur_score_test.png')

sc.stop()



