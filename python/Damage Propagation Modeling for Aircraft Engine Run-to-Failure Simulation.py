import findspark
findspark.init()
import pyspark
#sc = pyspark.SparkContext()

# Setup the column names of the training file
index_columns_names =  ["id","cycle"]
operational_settings_columns_names = ["setting"+str(i) for i in range(1,4)]
sensor_measure_columns_names =["s"+str(i) for i in range(1,22)]
sensor_measure_meancolumns_names =["a"+str(i) for i in range(1,22)]
sensor_measure_sdeccolumns_names =["sd"+str(i) for i in range(1,22)]

input_file_column_names = index_columns_names + operational_settings_columns_names + sensor_measure_columns_names

# And the name of the to be engineered target variable
dependent_var = ['rul']

from pyspark.sql import SQLContext
from pyspark.sql.types import *

sqlContext = SQLContext(sc)

scaledDF = sqlContext.read.parquet('/share/tedsds/output')
scaledDF.describe('setting1','s1','a1','sd1','rul', 's7', 's12', 's20').toPandas()

scaledDF.printSchema()

fraction = 1000.0 / scaledDF.count()
pf = scaledDF.select(dependent_var+operational_settings_columns_names+sensor_measure_columns_names).sample(fraction=fraction, withReplacement=False, seed=123456).toPandas()


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set() 

def labels(x, y, *axes):
    for axis in axes:
        if x: axis.set_xlabel(x)
        if y: axis.set_ylabel(y)

ax = sns.distplot(pf[["setting1"]])
labels("setting1", "p", ax)

pf_corr = pf.corr(method='pearson')
pf_corr

dims = (14,14)
fig, ax = plt.subplots(figsize=dims)
sns.heatmap(pf_corr,linewidths=.5);

sns.jointplot("rul", "s7", data=pf, kind='kde')  

g = sns.JointGrid(x="rul", y="s7", data=pf)  
g.plot_joint(sns.regplot, order=2)  
g.plot_marginals(sns.distplot)  

g = sns.PairGrid(pf[["rul", "s7", "s12", "s20"]], hue="rul", dropna=True)  
g.map_upper(sns.regplot)  
g.map_lower(sns.residplot)  
g.map_diag(plt.hist)  
for ax in g.axes.flat:  
    plt.setp(ax.get_xticklabels(), rotation=45)
g.set(alpha=0.5)  

fraction = 1000.0 / scaledDF.count()
pf2 = scaledDF.sample(fraction=fraction, withReplacement=False, seed=123456).toPandas()



g = sns.PairGrid(data=pf2,
                 x_vars=dependent_var,
                 y_vars=sensor_measure_columns_names + \
                        operational_settings_columns_names,
                 hue="id", size=2, aspect=2.5)
g = g.map(plt.plot, alpha=0.5)
g = g.set(xlim=(300,0))
#g = g.add_legend()


g = sns.PairGrid(data=pf2,
                 x_vars=dependent_var,
                 y_vars=sensor_measure_meancolumns_names,
                 hue="id", size=3, aspect=2.5)
g = g.map(plt.plot, alpha=0.5)
g = g.set(xlim=(300,0))


g = sns.PairGrid(data=scaledDF.toPandas(),
                 x_vars=dependent_var,
                 y_vars=sensor_measure_sdeccolumns_names,
                 hue="id", size=3, aspect=2.5)
g = g.map(plt.plot, alpha=0.5)
g = g.set(xlim=(400,0))
g = g.add_legend()


g = sns.pairplot(data=scaledDF.filter(scaledDF.id < 3).toPandas(),
                 x_vars=["setting1","setting2"],
                 y_vars=["s4", "s3", 
                         "s9", "s8", 
                         "s13", "s6"],
                 hue="id", aspect=2)

from pyspark.mllib.linalg import Vectors, DenseMatrix 
import numpy as np

def display_cm(m):
  a = m.toArray()
  #print(a)
  #print(m)
  row_sums = a.astype(np.float64).sum(axis=1).astype(np.float64)
  percentage_matrix = 100.0* a.astype(np.float64) / row_sums[:, np.newaxis]
  #print(percentage_matrix)
  plt.figure(figsize=(3, 3))
  dims = (8,8)
  fig, ax = plt.subplots(figsize=dims)
  sns.heatmap(percentage_matrix, annot=True,  fmt='.2f', xticklabels=['0' ,'1','2'], yticklabels=['0' ,'1','2']);
  plt.title('Confusion Matrix');

from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col
from pyspark.sql import SQLContext



def runmodel(model,data):
    # load the model
    ## trained with one data
    ## trained with all data
    lgregWithLBFGS = LogisticRegressionModel.load(sc, model)
    
    # load data for testset FD001DF 
    scaleddftest_FDDF = sqlContext.read.parquet(data)
    print("scaleddftest_FDDF count = %s" % scaleddftest_FDDF.count())
    
    # Index labels, adding metadata to the label column.
    #Fit on whole dataset to include all labels in index.
    indexer = StringIndexer(inputCol="label2", outputCol="indexedLabel")
    indexedDF = indexer.fit(scaleddftest_FDDF).transform(scaleddftest_FDDF)
    print("indexedDF count = %s" % indexedDF.count())
    labeledRDD = indexedDF.select(col("indexedLabel").alias("label"), col("scaledFeatures").alias("features")).map(lambda row:  LabeledPoint(row.label, row.features))
        
    # Compute raw scores on the test set
    predictionAndLabels = labeledRDD.map(lambda lp: (float(lgregWithLBFGS.predict(lp.features)), lp.label))
    # Instantiate metrics object
    metrics = MulticlassMetrics(predictionAndLabels)
    # Overall statistics
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    print("Summary Stats")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)
    # Statistics by class SF
    labels = labeledRDD.map(lambda lp: lp.label).distinct().collect()
    for label in sorted(labels):
        print("Class %s precision = %s" % (label, metrics.precision(label)))
        print("Class %s recall = %s" % (label, metrics.recall(label)))
        print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))
    # Weighted stats
    print("Weighted recall = %s" % metrics.weightedRecall)
    print("Weighted precision = %s" % metrics.weightedPrecision)
    print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
    print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
    print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)
    df_confusion = metrics.confusionMatrix()
    print (df_confusion)
    display_cm(df_confusion);

runmodel("/share/tedsds/savedmodelMulticlassMetricsExamplesaved","/share/tedsds/output")

runmodel("/share/tedsds/savedmodelallMulticlassMetricsExamplesaved","/share/tedsds/scaleddftest_FD001")

import numpy as np
        
class ROC_Point(object):
    def __init__(self,actual,predicted):
        ''' Method to initialize a ROC Curve with an observation'''
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive= 0
        self.false_negative= 0        
        if actual and predicted:
            self.true_positive = 1
        elif not actual and not predicted:
            self.true_negative = 1
        elif not actual and  predicted:
            self.false_positive= 1
        elif actual and not predicted:
            self.false_negative= 1   
 
    def ROC_add(self, ROC_Point):    
        ''' Method to aggregate ROC Curve'''
        self.true_positive  += ROC_Point.true_positive
        self.true_negative  += ROC_Point.true_negative
        self.false_positive += ROC_Point.false_positive
        self.false_negative += ROC_Point.false_negative
        return self
    def printme(self):
        print 'true_positive ... ' + str(self.true_positive) + '\n'
        print 'true_negative ... ' + str(self.true_negative) + '\n'
        print 'false_positive... ' + str(self.false_positive) + '\n'
        print 'false_negative ...' + str(self.false_negative) + '\n';
 
ROC_Point(True,True) 

def runmodel2(model,data):
    # load the model
    ## trained with one data
    ## trained with all data
    lgregWithLBFGS = LogisticRegressionModel.load(sc, model)
    
    # load data for testset FD001DF 
    scaleddftest_FDDF = sqlContext.read.parquet(data)
    print("scaleddftest_FDDF count = %s" % scaleddftest_FDDF.count())
    
    # Index labels, adding metadata to the label column.
    #Fit on whole dataset to include all labels in index.
    indexer = StringIndexer(inputCol="label2", outputCol="indexedLabel")
    indexedDF = indexer.fit(scaleddftest_FDDF).transform(scaleddftest_FDDF)
    print("indexedDF count = %s" % indexedDF.count())
    labeledRDD = indexedDF.select(col("indexedLabel").alias("label"), col("scaledFeatures").alias("features")).map(lambda row:  LabeledPoint(row.label, row.features))
        
    # Compute raw scores on the test set
    predictionAndLabels = labeledRDD.map(lambda lp: (float(lgregWithLBFGS.predict(lp.features)), lp.label))
    # Instantiate metrics object
    metrics = MulticlassMetrics(predictionAndLabels)
    # Overall statistics
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    print("Summary Stats")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)
    # Statistics by class SF
    labels = labeledRDD.map(lambda lp: lp.label).distinct().collect()
    for label in sorted(labels):
        print("Class %s precision = %s" % (label, metrics.precision(label)))
        print("Class %s recall = %s" % (label, metrics.recall(label)))
        print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))
    # Weighted stats
    print("Weighted recall = %s" % metrics.weightedRecall)
    print("Weighted precision = %s" % metrics.weightedPrecision)
    print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
    print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
    print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)
    df_confusion = metrics.confusionMatrix()
    display_cm(df_confusion)
    operating_threshold_bc = sc.broadcast(np.arange(0,1,.001))
    labelsAndPreds_Points = predictionAndLabels.map(lambda (label,prediction):  [ROC_Point(label==1,prediction>threshold)
    for threshold in operating_threshold_bc.value])
    labelsAndPreds_ROC_reduced = labelsAndPreds_Points.reduce( lambda l1,l2:  [ ROC_1.ROC_add(ROC_2) for ROC_1,ROC_2 in zip(l1,l2) ] )
    import matplotlib.pyplot as plt
    plt.plot(x,y,'ro')
    plt.plot(np.arange(0,1,.001),np.arange(0,1,.001), 'g-')
    plt.title('ROC Curve - TP rate vs. FP rate' )
    plt.show()

        

runmodel2("/share/tedsds/savedmodelallMulticlassMetricsExamplesaved","/share/tedsds/scaleddftrain_FD001")




