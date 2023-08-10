import py2neo
from py2neo import Graph

loadiris = '''
WITH "https://raw.githubusercontent.com/graphadvantage/neo4j-h2o-template/master/iris.csv" AS url
CALL apoc.load.csv(url,
  {
  header:true,
  mapping:{
    sepal_len:{type:'float'},
    sepal_wid:{type:'float'},
    petal_len:{type:'float'},
    petal_wid:{type:'float'},
    class:{type:'string'}
  }
}) YIELD map
CREATE (i:Iris)
SET i += map, i.uuid = apoc.create.uuid()
'''

# Some of these keyword arguments are unnecessary, as they are the default values.
graph = py2neo.Graph(bolt=True, host='localhost', user='neo4j', password='neo4j')

graph.run(loadiris)

import time
from IPython.display import display, HTML
import pandas as pd
from pandas import DataFrame
import py2neo
from py2neo import Graph

query1 = '''
MATCH (n:Iris) 
RETURN
n.class AS class,
n.petal_len AS petal_len,
n.petal_wid AS petal_wid,
n.sepal_len AS sepal_len,
n.sepal_wid AS sepal_wid,
n.uuid AS uuid
'''

# Some of these keyword arguments are unnecessary, as they are the default values.
graph = py2neo.Graph(bolt=True, host='localhost', user='neo4j', password='neo4j')

df = DataFrame(graph.data(query1))

#display(df)

df.style    .bar(subset=['petal_len'], color='#2980b9')    .bar(subset=['petal_wid'], color='#e74c3c')    .bar(subset=['sepal_len'], color='#27ae60')    .bar(subset=['sepal_wid'], color='#f1c40f')

import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set()

import h2o

h2o.init()

iris = h2o.H2OFrame(df)

iris.describe()

sns.set_context("notebook")
sns.pairplot(iris.as_data_frame(True), vars=["sepal_len", "sepal_wid", "petal_len", "petal_wid"], hue="class");

from h2o.estimators.kmeans import H2OKMeansEstimator

results = [H2OKMeansEstimator(k=clusters, init="Random", seed=2, standardize=True) for clusters in range(2,13)]
for estimator in results:
    estimator.train(x=iris.col_names[0:-1], training_frame = iris)

import math as math

def diagnostics_from_clusteringmodel(model):
    total_within_sumofsquares = model.tot_withinss()
    number_of_clusters = len(model.centers())
    number_of_dimensions = len(model.centers()[0])
    number_of_rows = sum(model.size())
    
    aic = total_within_sumofsquares + 2 * number_of_dimensions * number_of_clusters
    bic = total_within_sumofsquares + math.log(number_of_rows) * number_of_dimensions * number_of_clusters
    
    return {'Clusters':number_of_clusters,
            'Total Within SS':total_within_sumofsquares, 
            'AIC':aic, 
            'BIC':bic}

diagnostics = pd.DataFrame( [diagnostics_from_clusteringmodel(model) for model in results])
diagnostics.set_index('Clusters', inplace=True)
diagnostics.plot(kind='line');

clusters = 4
predicted = results[clusters-2].predict(iris)
iris["Predicted"] = predicted["predict"].asfactor()

sns.pairplot(iris.as_data_frame(True), vars=["sepal_len", "sepal_wid", "petal_len", "petal_wid"],  hue="Predicted");

iris

#Py2Neo Driver

import time

from string import Template

scores = iris[5:].as_data_frame()

df = DataFrame(scores)

#scores = df.to_list()

scores = df.to_json(orient = 'values')

#display(scores)

updateiris = Template(' WITH ${scores} AS records UNWIND records AS r WITH LOWER(r[0]) AS uuid, r[1] AS predicted MATCH (i:Iris {uuid: uuid}) SET i.predicted = predicted MERGE (c:Cluster {cluster: predicted}) MERGE (i)-[:MEMBER]->(c) ').substitute(locals())


graph = py2neo.Graph(bolt=True, host='localhost', user='neo4j', password='neo4j')

t0 = time.time()

graph.run(updateiris, scores = scores)

print(round((time.time() - t0)*1000,1), " ms elapsed time")

# Bolt Driver


import time

from string import Template

from neo4j.v1 import GraphDatabase, basic_auth, TRUST_ON_FIRST_USE, CypherError

driver = GraphDatabase.driver("bolt://localhost",
                              auth=basic_auth("neo4j", "neo4j"),
                              encrypted=False,
                              trust=TRUST_ON_FIRST_USE)

result = iris[5:].as_data_frame()

scores = DataFrame(result).to_json(orient = 'values')

#display(scores)

updateiris = Template(' WITH ${scores} AS records UNWIND records AS r WITH LOWER(r[0]) AS uuid, r[1] AS predicted MATCH (i:Iris {uuid: uuid}) SET i.predicted = predicted MERGE (c:Cluster {cluster: predicted}) MERGE (i)-[:MEMBER]->(c) ').substitute(locals())


session = driver.session()
t0 = time.time()
print("processing...")
result = session.run(updateiris, {"scores":scores})

for record in result:
    print(record)

summary = result.consume()
counters = summary.counters
print(summary)
print(counters)
print(round((time.time() - t0)*1000,1), " ms elapsed time")
print('-----------------')
session.close()




