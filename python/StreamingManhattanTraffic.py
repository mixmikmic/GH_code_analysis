# Where the web scraper script is outputting files
streamFilePath = "file:/C:/Users/JeffM/Documents/Projects/Spark Streaming/Data/"

# Number of seconds for Spark to ingest the data files
microBatchSeconds = 30

import os
import sys

# Set the path for the Spark folder
sparkPath = "C:/Users/JeffM/Documents/Spark/spark-2.0.2-bin-hadoop2.7"

os.environ['SPARK_HOME'] = sparkPath
sys.path.append(sparkPath + '/bin')
sys.path.append(sparkPath + '/python')
sys.path.append(sparkPath + '/python/pyspark' "C:/Spark/python/pyspark")
sys.path.append(sparkPath + '/python/pyspark/lib')
sys.path.append(sparkPath + '/python/pyspark/lib/pyspark.zip')
sys.path.append(sparkPath + '/python/pyspark/lib/py4j-0.10.3-src.zip')
sys.path.append("C:/Program Files/Java/jre1.8.0_111/bin")

import pyspark
from pyspark.streaming import StreamingContext

sc = pyspark.SparkContext("local[2]", "trafficData")  # Local[#] sets # of cores to use
ssc = StreamingContext(sc, microBatchSeconds)  # Seconds per micro-batch
sqlContext = pyspark.SQLContext(sc)

# Data manipulation
import pandas as pd
import numpy as np

# Timestamps
from datetime import datetime, timedelta

# City network and plots
import osmnx as ox, geopandas as gdp
import matplotlib.pyplot as plt

# To replace output plots for streaming
from IPython import display

# Sets intervals for updating the plot
import time

get_ipython().magic('matplotlib inline')

# Retrieving the network for Manhattan
manhattan = ox.graph_from_place('Manhattan, New York, USA', network_type='drive')

# Projecting the network for a top-down view
manhattan_projected = ox.project_graph(manhattan)

print('Non-projected:')
fig1 = ox.plot_graph(manhattan, fig_height=4.5, node_size=0, 
              node_alpha=1,edge_linewidth=0.2, dpi=100)

print('Projected:')
fig2 = ox.plot_graph(manhattan_projected, fig_height=4.5, node_size=0, 
              node_alpha=1,edge_linewidth=0.2, dpi=100)

# Importing only the fields we're interested in
dfNodes = pd.read_csv('trafficDataTemplate.txt',
            usecols=[0, 6, 11],
            names=['id', 'linkPoints', 'borough'])

# Filtering to Manhattan and dropping the borough column
dfNodes = dfNodes[dfNodes['borough'] == 'Manhattan']
dfNodes = dfNodes[['id', 'linkPoints']]

# Splitting the column containing all Lat/Long combinations
dfNodes['splitPoints'] = dfNodes['linkPoints'].apply(lambda x: x.split(' '))

# Reshaping our data frame to have a row for each Lat/Long point
idNodes = []
for x,y in zip(dfNodes['splitPoints'], dfNodes['id']):
    for a in np.arange(len(x)):
        idNodes.append((y, x[a]))
dfNodes = pd.DataFrame(idNodes, columns=['id', 'LatLong'])

dfNodes = dfNodes.replace('', np.NaN).dropna()

# Parsing Lat/Long into individual columns
# Longitude has an if statement since some records contain '-' for longitude
dfNodes['Latitude'] = dfNodes['LatLong'].apply(lambda x: x.split(';')[0])
dfNodes['Longitude'] = dfNodes['LatLong']     .apply(lambda x: x.split(';')[1] if len(x.split(';')) > 1 else None)

# Dropping incorrect longitude records and converting everything to floats
dfNodes = dfNodes[dfNodes['Longitude'] != '-'][['id', 'Latitude', 'Longitude']].astype(float)

# Obtaining the nearest nodes in the network from the Lat/Long points
nodes = []
for row in dfNodes.iterrows():
    nearest_node = ox.get_nearest_node(manhattan, (dfNodes['Latitude'].ix[row], 
                                                   dfNodes['Longitude'].ix[row]))
    nodes.append(nearest_node)
dfNodes['Node'] = nodes

# Removing duplicate nodes
dfNodes.drop_duplicates(subset='Node', inplace=True)

dfNodes.head()

# Reads files from the specified path
trafficStream = ssc.textFileStream(streamFilePath).map(lambda line: line.split(','))                    .filter(lambda line: line[11] == 'Manhattan')  # Removes other boroughs

# Puts specified columns into a temporary table
trafficStream.map(lambda line: (line[0], line[1], line[4], line[6], line[11]))     .foreachRDD(lambda rdd: rdd.toDF().registerTempTable("traffic"))

# Begins the streaming session
# Any other operations on the streaming data must come before this line
ssc.start()

while True:
    try:
        # Putting our streaming data from the temporary table into a data frame
        df = sqlContext.sql("select * from traffic").toPandas()
        df.columns = ['id', 'Speed', 'Date', 'linkPoints', 'Borough']

        # Removing records not in current year/month/day
        # Traffic data sometimes incorrectly includes old records signifying a sensor error
        df['Date'] = pd.to_datetime(df['Date'])  # Converting time to datetime
        df = df[df['Date'].dt.year == datetime.now().year]
        df = df[df['Date'].dt.month == datetime.now().month]
        df = df[df['Date'].dt.day == datetime.now().day]

        # Merging in the nodes
        df['id'] = df['id'].astype(float)  # Converting for the merge
        df = df.merge(dfNodes, on ='id')

        # Mapping the nodes to speed categories
        df['Speed'] = df['Speed'].astype(float)
        slow = []
        medium = []
        fast = []
        for row in df.iterrows():
            speed = df['Speed'].ix[row]
            node = df['Node'].ix[row]
    
            # Speeds in MPH
            if speed >= 40:
                fast.append(node)
            if speed < 40 and speed >= 20:
                medium.append(node)
            if speed < 20:
                slow.append(node)
        
        # Setting the node color based on the speed categories
        nc = ['g' if node in fast
              else ('y' if node in medium
              else ('r' if node in slow 
              else 'None'))
              for node in manhattan_projected.nodes()]

        # Timestamp
        EST = datetime.utcnow() - timedelta(hours=5)  # Calculating Eastern Standard Time
        print('Plot Generated:\n {:%Y-%m-%d \n %H:%M:%S} EST'.format(EST))

        # Plotting the map
        fig, ax = ox.plot_graph(manhattan_projected, fig_height=10, node_size=8, 
                                node_alpha=1,edge_linewidth=0.2, dpi=100, node_color=nc)
        # Sets the output
        display.display(ax)
        display.clear_output(wait=True)
        
        time.sleep(microBatchSeconds)  # Number of seconds between refreshes
    
    except KeyboardInterrupt:
        break

ssc.stop()

