from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql.functions import isnan, when, count, col

CSV_PATH = "/home/ds/notebooks/datasets/UCI_HAR/allData.csv"
CSV_ACTIVITY_LABEL_PATH = "/home/ds/notebooks/datasets/UCI_HAR/activity_labels.csv"
APP_NAME = "UCI HAR Random Forest Example"
SPARK_URL = "local[*]"
RANDOM_SEED = 141107
TRAINING_DATA_RATIO = 0.8
RF_NUM_TREES = 10
RF_MAX_DEPTH = 4
RF_NUM_BINS = 32

spark = SparkSession.builder.appName(APP_NAME).master(SPARK_URL).getOrCreate()
activity_labels = spark.read.options(inferschema = "true").csv(CSV_ACTIVITY_LABEL_PATH)
df = spark.read.options(inferschema = "true").csv(CSV_PATH)

# Confirm the dataframe shape is 10,299 rows by 562 columns
print(f"Dataset shape is {df.count():d} rows by {len(df.columns):d} columns.")

# Confirm that all feature columns are doubles via a list comprehension
# We're expecting 561 of 562 here, accounting for the labels column
double_cols = [col[0] for col in df.dtypes if col[1] == 'double']
print(f"{len(double_cols):d} columns out of {len(df.columns):d} total are type double.")

# Confirm there are no null values. We use the dataframe select method to build a 
# list that is then converted to a Python dict. This way it's easy to sum up the nulls.
null_counts = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) 
                         for c in df.columns]).toPandas().to_dict(orient='records')

print(f"There are {sum(null_counts[0].values()):d} null values in the dataset.")

# Generate our feature vector.
# Note that we're doing the work on the `df` object - we don't create new dataframes, 
# just add columns to the one we already are using.

# the transform method creates the column.

df = VectorAssembler(inputCols=double_cols, outputCol="features").transform(df)

df.select("_c0", "features").show(5)

# Build the training indexers / split data / classifier
# first we'll generate a labelIndexer
labelIndexer = StringIndexer(inputCol="_c0", outputCol="indexedLabel").fit(df)

# now generate the indexed feature vector
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(df)
    
# Split the data into training and validation sets (30% held out for testing)
(trainingData, testData) = df.randomSplit([TRAINING_DATA_RATIO, 1 - TRAINING_DATA_RATIO])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=RF_NUM_TREES)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Test Error = {(1.0 - accuracy):g}")
print(f"Accuracy = {accuracy:g}")

