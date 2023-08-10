print sc
print sqlContext
print sqlCtx

data_df = (sqlContext.read
           .format('com.databricks.spark.csv')
           .option("header", "true") # Use first line of all files as header
           .option("inferSchema", "true") # Automatically infer data types
           .load("skewdata-policy-new.csv")
           )

data_df.dtypes

data_df.show()

policyids = data_df.select('policyid').distinct()

policyids.show()

## Create a function the creates specified number of samples based on the sample size specified using fraction

def clusteredSamples(data,policies,policyid_sample_fraction,num_of_samples):
    
    #Initiate an emtpy sample list
    samples = []
    
    for n in range(0,num_of_samples):
        
        #Create a sample of the unique policy ids
        policyids_sample = policies.sample(withReplacement=False, fraction=policyid_sample_fraction)
    
        #Sample the data based on the sampled policyids
        sample = policyids_sample.join(data,on='policyid',how='inner')
        
        #Add the sample to the samples list
        samples.append(sample)
        
    #We will return a list of clustered samples
    return samples

sampleList = clusteredSamples(data_df,policyids,0.8,20)

sampleList

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

def runLinearRegression(samples):
    #initiate a result list
    samples_coefficients = []
    
    #Create a vector Assembler
    feature_columns = ['age']
    vectorAssembler = VectorAssembler(inputCols = feature_columns, outputCol = 'features_vector')
    
    #Create a linear regresson model
    lr = LinearRegression(featuresCol ='features_vector', 
                          labelCol = 'values',
                          predictionCol = 'predicted_values',
                          maxIter=5, 
                          elasticNetParam = 0.5,
                          solver="l-bfgs")
    for i in range(0,len(samples)):
        sample_df = samples[i]
        sample_df1 = vectorAssembler.transform(sample_df)
        
        #Fit the linear Regression model
        sample_lr = lr.fit(sample_df1)
        
        #Save the coefficients from the Regression model
        samples_coefficients.append(sample_lr.coefficients)
    
    #Return the list of coefficients from running glm on each sample set    
    return samples_coefficients

sampleCoefficients = runLinearRegression(sampleList)

sampleCoefficients



