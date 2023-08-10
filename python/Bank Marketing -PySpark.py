import findspark
findspark.init('directory_to_spark_installation')

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

data = spark.read.csv('bank.csv', 
                      header = True, 
                      inferSchema = True,
                      sep = ';')

data.printSchema()

data.columns

data.show()

data.groupBy('poutcome').count().show()

final_data = data.select('age', 'job', 'marital', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign', 'pdays', 'y')
final_data.describe().show()

final_data.printSchema()

# Converting string values to numerical column
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

JobIndexer = StringIndexer(inputCol = 'job', outputCol = 'JobIndex')
MaritalIndexer = StringIndexer(inputCol = 'marital', outputCol = 'MaritalIndex')
DefaultIndexer = StringIndexer(inputCol = 'default', outputCol = 'DefaultIndex')
HousingIndexer = StringIndexer(inputCol = 'housing', outputCol = 'HousingIndex')
LoanIndexer = StringIndexer(inputCol = 'loan', outputCol = 'LoanIndex')
LabelIndexer = StringIndexer(inputCol = 'y', outputCol = 'label')

# Using OneHotEncoder to avoid hierarchy in numerical value obtaied in above step
JobEncoder = OneHotEncoder(inputCol = 'JobIndex', outputCol = 'JobVec')
MaritalEncoder = OneHotEncoder(inputCol = 'MaritalIndex', outputCol = 'MaritalkVec')

# All the other columns have binary values(either Yes or No). So no need to hotencode

# Assemble everything together to be ("label","features") format
assembler = VectorAssembler(inputCols = ['age', 'JobVec', 'MaritalkVec', 'DefaultIndex', 'balance', 
                                    'HousingIndex', 'LoanIndex', 'duration', 'campaign', 'pdays'],
                           outputCol = 'features')

# Scaling the features
from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(inputCol = 'features', outputCol = 'scaledFeatures', withStd = True, withMean = False)

# Spliting the data into Training set and Test set
train_data, test_data =  final_data.randomSplit([0.8, 0.2])

# Defining Classifier Model
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

lr = LogisticRegression()
rcf = RandomForestClassifier(maxDepth = 10, numTrees = 300)

# Set Up the Pipeline
from pyspark.ml import Pipeline

# *************************** For Logistic Regressor ***************************
# pipeline = Pipeline(stages=[JobIndexer, MaritalIndexer, DefaultIndexer, HousingIndexer, 
#                             LoanIndexer, LabelIndexer, JobEncoder, MaritalEncoder, assembler, scaler, lr])

# *************************** For Random Forest Classifier ***************************
pipeline = Pipeline(stages=[JobIndexer, MaritalIndexer, DefaultIndexer, HousingIndexer, 
                            LoanIndexer, LabelIndexer, JobEncoder, MaritalEncoder, assembler, scaler, rcf])

# Fitting the model
model = pipeline.fit(train_data)

# Geting results on Test set
results = model.transform(test_data)

results.select('label', 'prediction').show()

# Model Evaluation

# Model selection is still in RDD phase. So We will use rdd here instead of spark dataframe. 
# This will be update in future versions of Spark
from pyspark.mllib.evaluation import MulticlassMetrics

predictionAndLabels = results.select('prediction', 'label').rdd

metrics = MulticlassMetrics(predictionAndLabels)

# Confusion Matrix
print('Confusion Matrix')
print(metrics.confusionMatrix().toArray())
print('Accuracy: %.2f'%metrics.accuracy)
print('Precision: %.2f'%metrics.weightedPrecision)



