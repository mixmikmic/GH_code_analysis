# First, let's verify that the SparkTK and daaltk libraries are installed
import sparktk
import daaltk

print "sparktk installation path = %s" % (sparktk.__path__)
print "daaltk installation path = %s" % (daaltk.__path__)

from sparktk import TkContext
tc = sparktk.TkContext(other_libs=[daaltk])

# Create a new frame by providing data and schema
data = [[2,"ab"],[1,"cd"],[7,"ef"],[1,"gh"],[9,"ij"],[2,"kl"],[0,"mn"],[6,"op"],[5,"qr"], [120, "outlier"]]
schema = [("data", float),("name", str)]

frame = tc.frame.create(data, schema)
frame.inspect()

# Consider the following frame containing two columns.
frame.inspect()

# DAAL KMeans model is trained using the frame from above
model = tc.daaltk.models.clustering.kmeans.train(frame, ["data"], k=2, max_iterations=20)
model

#call the modelto predict
predicted_frame = model.predict(frame, ["data"])
predicted_frame.inspect()

# Inspect HDFS directly using hdfsclient

import hdfsclient
from hdfsclient import ls, mkdir, rm, mv

try:
    rm("sandbox/myKMeansModel", recurse=True)
except:
    pass
model.save("sandbox/myKMeansModel")

restored = tc.load("sandbox/myKMeansModel")

restored

full_path = model.export_to_mar("sandbox/myKMeansModel.mar")

full_path

model.save

