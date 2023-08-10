print sc
print sqlContext
print sqlCtx

data_df = (sqlContext.read
                  .format('com.databricks.spark.csv')
                  .option("header", "true") # Use first line of all files as header
                  .option("inferSchema", "true") # Automatically infer data types
                  .load("skewdata.csv")
                  )

data_df.show()

import numpy as np
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import *

## Function to get confidence interval

def getConfidenceInterval(inputDataFrame,num_of_samples, left_quantile_fraction, right_quantile_fraction):
    #Simulate by sampling and calculating averages for each subsamples
    sample_means = np.empty([num_of_samples])
    for n in range(0,num_of_samples):
        sample_means[n] = (inputDataFrame.sample(withReplacement = True, fraction=1.0)
                   .selectExpr("avg(values) as mean")
                   .collect()[0]
                   .asDict()
                   .get('mean'))
            
    ## Sort the means
    sample_means.sort()
    
    ## Create a Pandas Dataframe from the numpy array
    sampleMeans_local_df = pd.DataFrame(sample_means)
    
    ## Create a Spark Dataframe from the pandas dataframe
    fields = [StructField("mean_values", DoubleType(), True)]
    schema = StructType(fields)
    sampleMeans_df = sqlContext.createDataFrame(sampleMeans_local_df, schema)
    
    ## Calculate the left_quantile and right_quantiles 
    sqlContext.registerDataFrameAsTable(sampleMeans_df, 'Guru_SampleMeansTable')
    quantiles_df = sqlContext.sql("select percentile(cast(mean_values as bigint),"
                                  "array("+str(left_quantile_fraction)+","+str(right_quantile_fraction)+")) as "
                                  "percentiles from Guru_SampleMeansTable")
    return quantiles_df

## Get 95% confidence interval in a two-tailed hypothesis testing
quantiles_df = getConfidenceInterval(data_df, 1000, 0.025, 0.975)

## We can now look at these percentiles and determine the critical region of sampling distribution
quantiles_df.show()



