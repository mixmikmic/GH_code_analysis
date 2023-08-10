sqlContext = SQLContext(sc)

# Connect to database 'sales' and read schema using all documents as schema sample size
cloudantdata = sqlContext.read.format("com.cloudant.spark").option("cloudant.host","examples.cloudant.com").option("schemaSampleSize", "-1").load("spark_sales")

# Print the schema that was detected
cloudantdata.printSchema()

# Cache the data
cloudantdata.cache()

# Count Data
print "Count is {0}".format(cloudantdata.count())

# Print Data

# Show 20 as default
cloudantdata.show()

# Show 5
cloudantdata.show(5)

# Show the rep field for 5
cloudantdata.select("rep").show(5)

# Run SparkSQL to get COUNTs and SUMs and do ORDER BY VALUE examples

# Register a temp table sales_table on the cloudantdata data frame
cloudantdata.registerTempTable("sales_table")

# Run SparkSQL to get a count and total amount of sales by rep
sqlContext.sql("SELECT rep AS REP, COUNT(amount) AS COUNT, SUM(amount) AS AMOUNT FROM sales_table GROUP BY rep ORDER BY SUM(amount) DESC").show(100)

# Run SparkSQL to get total amount of sales by month
sqlContext.sql("SELECT month AS MONTH, SUM(amount) AS AMOUNT FROM sales_table GROUP BY month ORDER BY SUM(amount) DESC").show()

# Graph the Monthly Sales  

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

pandaDF = sqlContext.sql("SELECT month AS MONTH, SUM(amount) AS AMOUNT FROM sales_table GROUP BY month ORDER BY SUM(amount) DESC").toPandas()
values = pandaDF['AMOUNT']
labels = pandaDF['MONTH']
plt.gcf().set_size_inches(16, 12, forward=True)
plt.title('Total Sales by Month')
plt.barh(range(len(values)), values)
plt.yticks(range(len(values)), labels)
plt.show()

# Filter, Count, Show, and Save Data

# Filter data for the rep 'Charlotte' and month of 'September'
filteredCloudantData = cloudantdata.filter("rep = 'Charlotte' AND month = 'September'")

# Count filtered data
print "Total Count is {0}".format(filteredCloudantData.count())

# Show filtered data
filteredCloudantData.show(5)

# Saving the amount, month, and rep fields from the filtered data...
# ...to a new Cloudant database 'sales_charlotte_september'
# NOTE: Remember to create the sales_charlotte_september database...
# ...in your Cloudant account AND replace ACCOUNT, USERNAME, and PASSWORD fields first!!
filteredCloudantData.select("amount","month","rep").write.format("com.cloudant.spark").option("cloudant.host","ACCOUNT.cloudant.com").option("cloudant.username","USERNAME").option("cloudant.password","PASSWORD").save("sales_charlotte_september")
print "Data is saved!"

