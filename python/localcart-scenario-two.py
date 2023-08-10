# run this cell
# jinja2 version 2.10 is required
get_ipython().system(' pip install jinja2 --user --upgrade')
# pixiedust version 1.1.7.1 is required
get_ipython().system(' pip install pixiedust --user --upgrade')
get_ipython().system(' pip install bokeh --user --upgrade')

import pixiedust
import pyspark.sql.functions as func
import pyspark.sql.types as types
import re
import json
import os
import requests  

raw_df = pixiedust.sampleData('https://raw.githubusercontent.com/ibm-watson-data-lab/localcart-at-index-conf/master/data/customers_orders1_opt.csv')

# Extract the customer information from the data set
# CUSTNAME: string, GenderCode: string, ADDRESS1: string, CITY: string, STATE: string, COUNTRY_CODE: string, POSTAL_CODE: string, POSTAL_CODE_PLUS4: int, ADDRESS2: string, EMAIL_ADDRESS: string, PHONE_NUMBER: string, CREDITCARD_TYPE: string, LOCALITY: string, SALESMAN_ID: string, NATIONALITY: string, NATIONAL_ID: string, CREDITCARD_NUMBER: bigint, DRIVER_LICENSE: string, CUST_ID: int,
customer_df = raw_df.select("CUST_ID", 
                            "CUSTNAME", 
                            "ADDRESS1", 
                            "ADDRESS2", 
                            "CITY", 
                            "POSTAL_CODE", 
                            "POSTAL_CODE_PLUS4", 
                            "STATE", 
                            "COUNTRY_CODE", 
                            "EMAIL_ADDRESS", 
                            "PHONE_NUMBER",
                            "AGE",
                            "GenderCode",
                            "GENERATION",
                            "NATIONALITY", 
                            "NATIONAL_ID", 
                            "DRIVER_LICENSE").dropDuplicates()

# append a column to the DataFrame for aggregations
customer_df = customer_df.withColumn("count", func.lit(1))
customer_df

# ---------------------------------------
# Cleanse age (enforce numeric data type) 
# ---------------------------------------

def getNumericVal(col):
    """
    input: pyspark.sql.types.Column
    output: the numeric value represented by col or None
    """
    try:
      return int(col)
    except ValueError:
      # age-33
      match = re.match('^age\-(\d+)$', col)
      if match:
        try:
          return int(match.group(1))
        except ValueError:    
          return None
      return None  

toNumericValUDF = func.udf(lambda c: getNumericVal(c), types.IntegerType())
customer_df = customer_df.withColumn("AGE", toNumericValUDF(customer_df["AGE"]))

# ------------------------------
# Derive gender from salutation
# ------------------------------
def deriveGender(col):
    """ input: pyspark.sql.types.Column
        output: "male", "female" or "unknown"
    """    
    if col in ['Mr.', 'Master.']:
        return 'male'
    elif col in ['Mrs.', 'Miss.']:
        return 'female'
    else:
        return 'unknown';
    
deriveGenderUDF = func.udf(lambda c: deriveGender(c), types.StringType())
customer_df = customer_df.withColumn("GENDER", deriveGenderUDF(customer_df["GenderCode"]))
customer_df.cache()

display(customer_df)

display(customer_df)

display(customer_df)

display(customer_df)

# Data subsetting: display age distribution for a specific generation
# (Chart type: histogram, Chart Options > Values: AGE)
# to change the filter condition remove the # sign 
condition = "GENERATION = 'Baby_Boomers'"
#condition = "GENERATION = 'Gen_X'"
#condition = "GENERATION = 'Gen_Y'"
#condition = "GENERATION = 'Gen_Z'"
display(customer_df.filter(condition))

display(customer_df)

condition = "COUNTRY_CODE = 'US'"
us_customer_df = customer_df.filter(condition)

display(us_customer_df)

display(us_customer_df)

display(us_customer_df)

# Load median income information for all US ZIP codes from a public source
income_df = pixiedust.sampleData('https://apsportal.ibm.com/exchange-api/v1/entries/beb8c30a3f559e58716d983671b70337/data?accessKey=1c0b5b6d465fefec1ab529fde04997af')

# ------------------------------
# Helper: Extract ZIP code
# ------------------------------
def extractZIPCode(col):
    """ input: pyspark.sql.types.Column containing a geo code, like '86000US01001'
        output: ZIP code
    """
    m = re.match('^\d+US(\d\d\d\d\d)$',col)
    if m:
        return m.group(1)
    else:
        return None    
    
getZIPCodeUDF = func.udf(lambda c: extractZIPCode(c), types.StringType())
income_df = income_df.select('GEOID', 'B19049e1').withColumnRenamed('B19049e1', 'MEDIAN_INCOME_IN_ZIP').withColumn("ZIP", getZIPCodeUDF(income_df['GEOID']))
income_df

us_customer_df = us_customer_df.join(income_df, us_customer_df.POSTAL_CODE == income_df.ZIP, 'left_outer').drop('GEOID').drop('ZIP')

display(us_customer_df)

# Extract sales information from raw data set
# 
sales_df = raw_df.select("CUST_ID", 
                         "CITY", 
                         "STATE", 
                         "COUNTRY_CODE", 
                         "GenderCode",
                         "GENERATION",
                         "AGE",
                         "CREDITCARD_TYPE",
                         "ORDER_ID",
                         "ORDER_TIME",
                         "FREIGHT_CHARGES",
                         "ORDER_SALESMAN",
                         "ORDER_POSTED_DATE",
                         "ORDER_SHIP_DATE",
                         "ORDER_VALUE",
                         "T_TYPE",
                         "PURCHASE_TOUCHPOINT",
                         "PURCHASE_STATUS",
                         "ORDER_TYPE",
                         "Baby Food",
                         "Diapers",
                         "Formula",
                         "Lotion", 
                         "Baby wash",
                         "Wipes",
                         "Fresh Fruits",
                         "Fresh Vegetables",
                         "Beer",
                         "Wine",
                         "Club Soda",
                         "Sports Drink",
                         "Chips",
                         "Popcorn",
                         "Oatmeal",
                         "Medicines",
                         "Canned Foods",
                         "Cigarettes",
                         "Cheese",
                         "Cleaning Products",
                         "Condiments",
                         "Frozen Foods",
                         "Kitchen Items",
                         "Meat",
                         "Office Supplies",
                         "Personal Care",
                         "Pet Supplies",
                         "Sea Food",
                         "Spices").dropDuplicates()

# add column containing the numeric value 1. it will be used to perform aggregations
sales_df = sales_df.withColumn("count", func.lit(1))

# ---------------------------------------
# Cleanse age (enforce numeric data type) 
# ---------------------------------------

def getNumericVal(col):
    """
    input: pyspark.sql.types.Column
    output: the numeric value represented by col or None
    """
    try:
      return int(col)
    except ValueError:
      # age-33
      match = re.match('^age\-(\d+)$', col)
      if match:
        try:
          return int(match.group(1))
        except ValueError:    
          return None
      return None  

toNumericValUDF = func.udf(lambda c: getNumericVal(c), types.IntegerType())
sales_df = sales_df.withColumn("AGE", toNumericValUDF(sales_df["AGE"]))

# ------------------------------
# Derive gender from salutation
# ------------------------------
def deriveGender(col):
    """ input: pyspark.sql.types.Column
        output: "male", "female" or "unknown"
    """    
    if col in ['Mr.', 'Master.']:
        return 'male'
    elif col in ['Mrs.', 'Miss.']:
        return 'female'
    else:
        return 'unknown';
    
deriveGenderUDF = func.udf(lambda c: deriveGender(c), types.StringType())
sales_df = sales_df.withColumn("GENDER", deriveGenderUDF(sales_df["GenderCode"]))

# ------------------------------
# get date column as string
# ------------------------------
def getDateString(datetime_col, format_string):
    """ input: pyspark.sql.types.Column
        input: string a strftime format string (https://docs.python.org/2/library/time.html#time.strftime)
        output: a formatted date string
    """    
    if format_string is None:
        format_string = '%d/%m/%Y'
    return datetime_col.strftime(format_string)

# append columns to data set:
#  - add ORDER_S_DATE (string representation of the order date - to address a PixieDust limitation)
#  - add ORDER_DATE_YEAR (string representation of the order date year: YYYY)
#  - add ORDER_DATE_MONTH (string representation of the order date month: YYYY-MM)

getDateStringUDF = func.udf(lambda c: getDateString(c, None), types.StringType())
sales_df = sales_df.withColumn("ORDER_DATE_S", getDateStringUDF(sales_df["ORDER_TIME"]))

getYearStringUDF = func.udf(lambda c: getDateString(c,'%Y'), types.StringType())
sales_df = sales_df.withColumn("ORDER_DATE_YEAR", getYearStringUDF(sales_df["ORDER_TIME"]))

getMonthStringUDF = func.udf(lambda c: getDateString(c,'%Y-%m'), types.StringType())
sales_df = sales_df.withColumn("ORDER_DATE_MONTH", getMonthStringUDF(sales_df["ORDER_TIME"]))

# cache the DataFrame to speed up analysis
sales_df.cache()

display(sales_df.groupBy("COUNTRY_CODE").count())

# Apply optional filter 1: restrict geographic location to USA
sales_df = sales_df.filter("COUNTRY_CODE = 'US'")

display(sales_df.groupBy("T_TYPE").count())

# apply optional transaction status filter. Subsequent analysis is limited to these types of transactions
txn_sales_df = sales_df.filter("T_TYPE = 'Complete'")
#txn_sales_df = sales_df.filter("T_TYPE = 'Cancelled'")
#txn_sales_df = sales_df.filter("T_TYPE = 'Abandoned'")
#txn_sales_df = sales_df.filter("T_TYPE = 'In-Progress'")

display(txn_sales_df)

display(txn_sales_df)

display(txn_sales_df)

# only inspect completed transactions
classify_df = sales_df.filter("T_TYPE = 'Complete'")
# test data set contains synthetic outliers. remove them
classify_df = classify_df.filter("ORDER_VALUE < 9999999")

display(classify_df)

# identify metadata columns in this data set
metadata_columns = [
                    'CUST_ID', 
                    'CITY', 
                    'count',
                    'STATE', 
                    'COUNTRY_CODE', 
                    'GenderCode', 
                    'GENDER',
                    'GENERATION',
                    'AGE',
                    'CREDITCARD_TYPE',
                    'ORDER_DATE_S',
                    'ORDER_DATE_YEAR',
                    'ORDER_DATE_MONTH',
                    'ORDER_ID',
                    'ORDER_TIME',
                    'FREIGHT_CHARGES',
                    'ORDER_SALESMAN',
                    'ORDER_POSTED_DATE',
                    'ORDER_SHIP_DATE',
                    'ORDER_VALUE',
                    'T_TYPE',
                    'PURCHASE_TOUCHPOINT',
                    'PURCHASE_STATUS',
                    'ORDER_TYPE',
                    'GENERATION'
                    ]

# identify item category columns
category_columns = []
for data_column in sales_df.columns:
    if data_column not in metadata_columns:
        category_columns.append(data_column)

exprs = {x: "sum" for x in category_columns}
category_count_df = sales_df.groupBy("ORDER_DATE_MONTH").agg(exprs).alias("col")

display(category_count_df)

fresh_veggies_df = classify_df.filter(classify_df["Fresh Vegetables"] > 0)
display(fresh_veggies_df)

transaction_type = 'Abandoned'

abandoned_df = sales_df.filter("T_TYPE = 'Complete' OR T_TYPE = '" + transaction_type + "'")

display(abandoned_df)

# create new data set comprising of all <transaction_type> sales transactions
tx_type_sales_df = sales_df.filter("T_TYPE = '" + transaction_type + "'")
# test data set contains synthetic outliers. remove them
tx_type_sales_df = tx_type_sales_df.filter("ORDER_VALUE < 9999999")

display(tx_type_sales_df)

display(tx_type_sales_df)

exprs = {x: "sum" for x in category_columns}
item_count_df = tx_type_sales_df.groupBy("ORDER_DATE_MONTH").agg(exprs)

display(item_count_df)

