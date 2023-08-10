import sys
get_ipython().system('{sys.executable} -m pip install plotly')
get_ipython().system('{sys.executable} -m pip install geopy')
get_ipython().system('{sys.executable} -m pip install cufflinks')

import matplotlib.cm as cm
import matplotlib as mpl
from geopy.geocoders import Nominatim
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode

import pyspark
from pyspark.sql import SQLContext
import pyspark.sql.functions
from pyspark.sql.functions import avg
import time

import plotly.figure_factory as ff
import plotly.graph_objs as go
import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
cf.go_offline()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Setting Spark Context
sc = pyspark.SparkContext('local[*]')
sqlContext=SQLContext(sc)

#Seeing the current directory
get_ipython().system('ls')

#Loading all the Necessary datasets
df_pop = sqlContext.read.format("com.databricks.spark.csv")         .options(header = "true", inferschema = "true")         .load("energy-pop-exposure-nuclear-plants-locations_plants.csv")
df_reactors = sqlContext.read.format("com.databricks.spark.csv")             .options(header = "true", inferschema = "true")             .load("energy-pop-exposure-nuclear-plants-locations_reactors.csv")
df_measurements = sqlContext.read.format("com.databricks.spark.csv")         .options(header = "true", inferschema = "true")         .load("measurements.csv")

#Registering the dataset as a SQL Table
df_reactors.registerTempTable("reactors")

#Checking the schema of our dataset
df_reactors.printSchema()

#Querying to get the total power generated country-wise

df_total_power = sqlContext.sql("SELECT Country, SUM(Totalpower) as total_power FROM reactors GROUP BY Country ORDER BY Country").toPandas()
df_total_power.head()

#Importing pycountry package to get the country codes

import pycountry as pc

country_dict= {}

for country in pc.countries:
    country_dict[country.name.upper()] = country.alpha_3

df_total_power['Code'] = df_total_power['Country'].apply(lambda x : country_dict.get(x))

#Country Codes for all the Countries

df_total_power.head()

#Constructing a choropleth map

data = dict(
        type = 'choropleth',
        locations = df_total_power['Code'],
        z = df_total_power['total_power'],
        text = df_total_power['Country'],
        colorbar = {'title' : 'Power Generated'},
      ) 

layout = dict(
    title = 'Power Generated',
    geo = dict(
        showframe = False,
        projection = {'type':'Mercator'}
    )
)

choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)

df_reactor_count = pd.DataFrame(df_reactors[['Status']].toPandas()['Status'].value_counts())
df_reactor_count['label'] = df_reactor_count.index
#Status of Reactors

#Constructing a Pie-Chart to see the total type of reactors

df_reactor_count.iplot(kind='pie',labels='label',values='Status', title = 'Status of Reactors')

#Plotting the number of reactors per year

df_reactors[['Year_Built']].toPandas()['Year_Built'].value_counts().sort_index().iplot(title = 'Number of reactors built per year')

#Finding the total number of power generated per year all over the world

output_names = [name for name in df_reactors.columns if (name[0:3] == "Ref")]

df_ref = df_reactors.select(output_names).toPandas()
df_ref.head()

newCol = []
for col in df_ref.columns:
    newCol.append(int(col[4:]))
    
df_ref.columns = newCol
df_ref.sum().sort_index().iplot(title = 'Power Generated per year all over the world')

output_names.append('Country')
df_ref_with_country = df_reactors.select(output_names).toPandas()

df_ref_with_country.head()

df_ref_with_country = df_ref_with_country.groupby('Country').sum().T

#Plotting power generated country-wise
newIndex = []
for index in df_ref_with_country.index:
    newIndex.append(int(index[4:]))
    
df_ref_with_country.index = newIndex

df_ref_with_country = df_ref_with_country.sort_index(ascending=True)

df_ref_with_country.iplot(title = 'Power Generated Country-wise')

from pyspark.sql.functions import *

df_pop.registerTempTable("plants_pop")

x = sqlContext.sql("select Country, sum(NumReactor) AS NumReactor from plants_pop group by Country")

x.show()

pandas_data= x.toPandas()

pandas_data.head()

pandas_data.plot(x='Country',y='NumReactor', kind='bar', figsize=(20,10), sort_columns=True)
plt.show()

lat_long = sqlContext.sql('select Region, Country, NumReactor,                             Plant, NumReactor, Latitude, Longitude from plants_pop')

lat_long_pandas = lat_long.toPandas()

lat_long_pandas.head()

lat_list = lat_long_pandas.Latitude.tolist()
long_list = lat_long_pandas.Longitude.tolist()

data = [ dict(
        type = 'scattergeo' ,
        lat = lat_list,
        lon = long_list,
        mode = 'markers',
        )]

layout = dict(
    title = 'Reactor Locations',
    )

plotly.offline.init_notebook_mode(connected = True)
fig = dict( data = data, layout = layout)
plotly.offline.iplot(fig)

#Cleaning the measurements data
df_measurements=df_measurements.withColumn("Unit", regexp_replace(col("Unit"), "uSv/hr", "usv"))
df_measurements=df_measurements.withColumn("Unit", regexp_replace(col("Unit"), "uSv/h", "usv"))
df_measurements=df_measurements.withColumn("Unit", regexp_replace(col("Unit"), "usv/hr", "usv"))
df_measurements=df_measurements.withColumn("Unit", regexp_replace(col("Unit"), "uSv", "usv"))
df_measurements=df_measurements.withColumn("Unit", regexp_replace(col("Unit"), "CPM", "cpm"))
df_measurements=df_measurements.withColumn("Unit", regexp_replace(col("Unit"), " cpm", "cpm"))
df_measurements=df_measurements.withColumn("Unit", regexp_replace(col("Unit"), "Cpm", "cpm"))
df_measurements=df_measurements.withColumn("Unit", regexp_replace(col("Unit"), "cpm ", "cpm"))
df_measurements=df_measurements.withColumn("Unit", regexp_replace(col("Unit"), "microsievert", "usv"))


df_ms=df_measurements.where((col("Unit") == "cpm") | (col("Unit") == "usv"))
df_ms= df_measurements.withColumn("Longitude", df_measurements["Longitude"].cast("double"))


df_ms=df_ms.select("Captured Time","Latitude","Longitude","Value","Unit")
df_ms=df_ms.withColumnRenamed('Captured Time', 'time')

df_ms.printSchema()

df_ms.where("Unit=='usv'").show(100)

#Function to convert usv to cpm (these are radiation units)
#Just maintaining only one unit (standardizing)
def usvtocpm(v):
    return (v*2520/21)

from pyspark.sql.functions import udf
cpm_udf = udf(usvtocpm)

#Applying the above function to standardize the values
df_val=df_ms.withColumn("Value",               when(df_ms["Unit"] == 'usv', cpm_udf("Value")).otherwise(df_ms["Value"])).where((col("Unit") == "cpm") | (col("Unit") == "usv"))

#looking at the schema 
df_val.printSchema()

df_val.show(5)

#Converting time which is in String format, to timestamp format


df_val = df_val.withColumn("parsed_time", unix_timestamp("time", "yyyy-MM-dd HH:mm:ss")
    .cast("double")
    .cast("timestamp"))

df_val.show(5)

df_val.printSchema()

#df_val.show()
df_val = df_val.drop('time')

df_val.show()

#Getting the individual year and month columns

df_val_yr_month = df_val.select('Latitude', 'Longitude', 'Value', 'Unit',year('parsed'), month('parsed'))#.alias('year', 'month')
df_val_yr_month.show(100)

df_val_filtered_year = df_val_yr_month.filter(df_val_yr_month)

df_val.where(col("time").isNull()).count()

df_val.where(col("Latitude").isNull()).count()

df_val.where(col("Longitude").isNull()).count()

#df_val.where(col("Value").isNull()).count()

df_val.where(col("Unit").isNull()).count()

df_val.where(col("time").isNull()).show(10000)

df_val.show(20)

#Dropping the NaN values in time
df_val_None = df_val.na.drop(subset = ["time"])

d=df_val_None.withColumn("time", ts_udf(df_val_None.time))

d.printSchema()

#Loading the cleaned data
df_all_measure = sqlContext.read.format("com.databricks.spark.csv").option("header", "false").load("Downloads/measurement_cleaned.csv/*.csv")

df_all_measure.show(20)

#Giving proper column names
df_all_measure = df_all_measure.withColumnRenamed("_c0", "Latitude").withColumnRenamed("_c1", "Longitude").withColumnRenamed("_c2", "RadiationLevel").withColumnRenamed("_c3", "Year").withColumnRenamed("_c4", "Month")

#Registering the cleaned data as a SQL Table
df_all_measure.registerTempTable("final_data")

#Querying to get the avg radiation level year and month wise
scaled_df_all_measure = sqlContext.sql("SELECT Year, Month, AVG(RadiationLevel) AS AvgRadiation FROM final_data GROUP BY Year, Month ORDER BY Year, Month")

#Converting Year and AvgRadiation to Int from String type
from pyspark.sql.types import IntegerType

scaled_df_all_measure = scaled_df_all_measure.withColumn("Year", scaled_df_all_measure["Year"].cast(IntegerType()))
scaled_df_all_measure = scaled_df_all_measure.drop('Month')

scaled_df_all_measure.dtypes

#Preliminary steps to build a Linear Model
from pyspark.sql import SparkSession
spark = SparkSession.builder    .master("local")    .appName("Linear Regression Model")    .config("spark.executor.memory", "1gb")    .getOrCreate()

# Import `DenseVector`
from pyspark.ml.linalg import DenseVector

# Define the `input_data` 
input_data = scaled_df_all_measure.rdd.map(lambda x: (DenseVector(x[0:]), x[1] ))

# Replace `df` with the new DataFrame
scaled_df_all_measure = spark.createDataFrame(input_data, ["features", "label"])

#Scaling both the features and label so that they lie within the same range

# Import `StandardScaler` 
from pyspark.ml.feature import StandardScaler

# Initialize the `standardScaler`
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")

# Fit the DataFrame to the scaler
scaler = standardScaler.fit(scaled_df_all_measure)

# Transform the data in `df` with the scaler
scaled_df = scaler.transform(scaled_df_all_measure)

# Inspect the result
scaled_df.take(2)

# Split the data into train and test sets
train_data, test_data = scaled_df.randomSplit([.8,.2],seed=1234)

# Import `LinearRegression`
from pyspark.ml.regression import LinearRegression

# Initialize `lr`
lr = LinearRegression(labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the data to the model
linearModel = lr.fit(train_data)

# Generate predictions
predicted = linearModel.transform(test_data)

# Extract the predictions and the "known" correct labels
predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
labels = predicted.select("label").rdd.map(lambda x: x[0])

# Zip `predictions` and `labels` into a list
predictionAndLabel = predictions.zip(labels).collect()

# Print out first 5 instances of `predictionAndLabel` 

del predictionAndLabel[11]

x_values = [predictionAndLabel[i][0] for i in range(len(predictionAndLabel)-4)]
y_values = [predictionAndLabel[i][1] for i in range(len(predictionAndLabel)-4)]


import matplotlib.pyplot as plt
plt.figure(figsize =(10,10))
plt.plot(x_values,'r')
plt.plot(y_values,'b')
plt.title("Predicted Radiation levels")
plt.xlabel("Number of Months after October 2017")
plt.ylabel("Radiation level in cpm")
plt.show()

#Evaluating the fit of our Linear Regression Model by looking at the root mean squared error
linearModel.summary.rootMeanSquaredError



