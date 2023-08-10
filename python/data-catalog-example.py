# First, let's verify that the sparktk libraries are installed
import sparktk
print "sparktk installation path = %s" % (sparktk.__path__)

# This notebook assumes you have already created a credentials file.
# Enter the path here to connect to ATK
from sparktk import TkContext
tc = TkContext()

# Create a new frame by uploading rows
data = [ [4.9,1.4,0], 
        [4.7,1.3,0], 
        [4.6,1.5,0], 
        [6.3,4.9,1],
        [6.1,4.7,1], 
        [6.4,4.3,1], 
        [6.6,4.4,1], 
        [7.2,6.0,2],
        [7.2,5.8,2], 
        [7.4,6.1,2], 
        [7.9,6.4,2]]

schema = [('Sepal_Length', float),
          ('Petal_Length', float),
          ('Class', int)]
frame = tc.frame.create(data, schema)

# Consider the following frame containing three columns.
frame.inspect()

# Create a new model and train it
model = tc.models.classification.naive_bayes.train(frame, ['Sepal_Length', 'Petal_Length'], 'Class')

# Export the trained model to MAR format
model.export_to_mar("hdfs://nameservice1/user/vcap/example_naive_bayes_model.mar")

# Import Data Catalog client module from tap_catalog
from tap_catalog import DataCatalog

# Create an instance of Data Catalog
## data_catalog = DataCatalog('TAP_DOMAIN_URI', 'TAP_USERNAME', 'TAP_PASSWORD') # For Scripting purposes
data_catalog = DataCatalog()

# Add an entry to Data Catalog
data_catalog.add("/user/vcap/example_naive_bayes_model.mar")

# Inspect HDFS directly using hdfsclient

import hdfsclient
from hdfsclient import ls, mkdir, rm, mv

ls("/user/vcap/example_naive_bayes_model.mar")

# Cleanup the file from HDFS
## (This does not delete from data catalog. Remember to delete it from the Data Catalog UI)
rm("/user/vcap/example_naive_bayes_model.mar")

