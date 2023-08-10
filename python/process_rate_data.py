import pandas as pd
from pyspark.sql import SparkSession
spark = (SparkSession.builder 
    .appName("Concat Rates") 
    .getOrCreate())

from pyspark.sql.types import *
from pyspark.sql.functions import *

rates2014 = (spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
             .load('s3://ryan-afa/input/rates-2014.csv', inferSchema=True, header=True)
             .cache())

plans2014 = (spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
             .load('s3://ryan-afa/input/plan-2014.csv', inferSchema=True, header=True)
             .cache())
rates2017 = (spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
             .load('s3://ryan-afa/input/rates-2017.csv', inferSchema=True, header=True)
             .cache())
plans2017 = (spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
             .load('s3://ryan-afa/input/plan-2017.csv', inferSchema=True, header=True)
             .cache())

rates2016 = (spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
             .load('s3://ryan-afa/input/rates-2016.csv', inferSchema=True, header=True)
             .cache())
plans2016 = (spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
             .load('s3://ryan-afa/input/plan-2016.csv', inferSchema=True, header=True)
             .cache())

rates2015 = (spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
             .load('s3://ryan-afa/input/rates-2015.csv', inferSchema=True, header=True)
             .cache())
plans2015 = (spark.read.format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
             .load('s3://ryan-afa/input/plan-2015.csv', inferSchema=True, header=True)
             .cache())



def getMeans(rates, plans):
    
    #missing rate data was filled with 9999 -- so we get rid of those rows 
    rates = rates.filter(rates.IndividualRate  < 9999)
    
    # to get the metal level for each plan, we need to join the rates data
    # the plan attributes data set. But first we reformat the planId column
    # so we can do the join, filter out dental plans, and get unique combinations
    # of the variables we're interested in.
    plans = plans.withColumn('PlanId', substring_index(plans.PlanId, '-', 1))
    plans = plans.filter(plans.DentalOnlyPlan != "Yes")
    plans = plans.dropDuplicates(['PlanId', 'StateCode', 'MetalLevel'])
    
    #join the data sets
    merged = rates.join(plans.select('StateCode', 'PlanId', 'MetalLevel'), on=['StateCode', 'PlanId'])
    
    #compute the means and standard deviations 
    means = (merged.groupby(['Age', 'MetalLevel', 'StateCode'])
             .agg(mean(merged.IndividualRate), stddev(merged.IndividualRate)))
    
    #rename the columns we just computed so they're friendlier. 
    return (means
            .withColumnRenamed('avg(IndividualRate)', 'avg_IndividualRate')
            .withColumnRenamed('stddev_samp(IndividualRate', 'std_IndividualRate')).cache()

merged2014 = getMeans(rates2014, plans2014)
merged2017 = getMeans(rates2017, plans2017)
merged2015 = getMeans(rates2015, plans2015)
merged2016 = getMeans(rates2016, plans2016)

(merged2017
 .coalesce(1)
 .write
 .format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
 .save('s3://ryan-afa/output/averages-2017.csv',  header=True))

(merged2016
 .coalesce(1)
 .write
 .format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
 .save('s3://ryan-afa/ouput/averages-2016.csv',  header=True))

(merged2014
 .coalesce(1)
 .write
 .format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
 .save('s3://ryan-afa/output/averages-2014.csv',  header=True))

(merged2015
 .coalesce(1)
 .write
 .format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
 .save('s3://ryan-afa/output/averages-2015.csv',  header=True))

def getPercChange(rates1, rates2, suffixes):
    

    #add suffixes to the column names so we can distinguish them after we join the dataframes
    firstYearAvg =  'avg_IndividualRate' + suffixes[0]
    secondYearAvg = 'avg_IndividualRate' + suffixes[1]
    firstYearStd = 'std_IndividualRate' +  suffixes[0]
    secondYearStd = 'std_IndividualRate' +  suffixes[1]
    rates1 = (rates1.withColumnRenamed('avg_IndividualRate', firstYearAvg)
              .withColumnRenamed('stddev_samp(IndividualRate)', firstYearStd))
    rates2 = (rates2.withColumnRenamed('avg_IndividualRate', secondYearAvg)
             .withColumnRenamed('stddev_samp(IndividualRate)', secondYearStd))
    
    #join the rates for consecutive years
    ratesJoined = rates1.join(rates2, on=['StateCode', 'Age', 'MetalLevel'])
    
    #for each rate column, calculate the percentage change in rate between the years, and add it to the dataframe. 
    ratesJoined = ratesJoined.withColumn(
            'percentChange', ((ratesJoined[secondYearAvg] - ratesJoined[firstYearAvg]) / ratesJoined[firstYearAvg]) * 100)
    return ratesJoined


change1415 = getPercChange(merged2014, merged2015, ['_2014', '_2015'])
change1516 = getPercChange(merged2015, merged2016, ['_2015', '_2016'])
change1617 = getPercChange(merged2016, merged2017, ['_2016', '_2017'])


(change1415
 .coalesce(1)
 .write
 .format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
 .save('s3://ryan-afa/output/changes-1415.csv',  header=True))

(change1516
 .coalesce(1)
 .write
 .format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
 .save('s3://ryan-afa/ouput/changes-1516.csv',  header=True))

(change1617
 .coalesce(1)
 .write
 .format("org.apache.spark.sql.execution.datasources.csv.CSVFileFormat")
 .save('s3://ryan-afa/output/changes-1617.csv',  header=True))



