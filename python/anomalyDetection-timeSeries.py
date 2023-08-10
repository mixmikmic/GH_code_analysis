get_ipython().magic('matplotlib inline')
import json, csv, sys, datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer

from sklearn.decomposition import TruncatedSVD, PCA
from mpl_toolkits.mplot3d import Axes3D

from sklearn import cluster

from sklearn.neighbors import NearestNeighbors
import matplotlib.font_manager

from sklearn import svm
from sklearn.model_selection import train_test_split


with open('../dataset/sample_first_50000.json', 'r') as file:
	i = 0
	j = 0
	# TODO: change list-based stuff to dict
	data = {}
	ipList = []
	# iterate over each logged line
	for line in file:
		newItem = {}
		try:
			jsonData = json.loads(line)
		except:
			print "\nLine {0} is not in JSON format".format(i)
			i += 1
			j += 1
			continue

		if 'data' in jsonData:
			timestamp = str(jsonData['data']['event_timestamp'])
			hour_of_the_day = (float(timestamp[11:13]) + float(timestamp[14:16])/60 + float(timestamp[17:19])/3600)
			# 0 - Monday; 6 - Sunday
			day_of_the_week = datetime.datetime(int(timestamp[:4]), 				int(timestamp[5:7]), int(timestamp[8:10]), int(timestamp[11:13]), 				int(timestamp[14:16]), int(timestamp[17:19])).weekday()
			newItem["timestamp"] = timestamp
			newItem["hour_of_the_day"] = hour_of_the_day
			newItem["day_of_the_week"] = day_of_the_week
			featuresFromData = ["client_user", "client_host", "client_ip", "client_program", "CONNECT_DATA_INSTANCE_NAME", "service_name"]
			for feature in featuresFromData:
				if feature in jsonData['data']:
					newItem[feature] = str(jsonData['data'][feature])
				else:
					newItem[feature] = ""

		if 'metadata' in jsonData:
			featuresFromMetadata = ["timestamp","oracle_sid", "hostname"]
			for feature in featuresFromMetadata:
				if feature in jsonData['metadata']:
					newItem[feature] = str(jsonData['metadata'][feature])
				else:
					newItem[feature] = ""

		# ignore cases where data is incomplete/very little to analyse
		if len(newItem) <= 2:
			continue

		else:
			data[i] = newItem
			ipList.append(newItem["client_ip"])
		# increment item number within the data
		i += 1

if j > 0:
	print "Could not store {0} lines due to invalid format".format(j)

fieldNames = ['timestamp', 'hour_of_the_day', 'day_of_the_week', 'client_user', 'client_host', 'client_ip', 'client_program', 'CONNECT_DATA_INSTANCE_NAME', 'service_name', 'oracle_sid', 'hostname']

with open('../dataset/preprocessed_first_50000.csv', 'w') as csvFile:
	writer = csv.DictWriter(csvFile, fieldnames=fieldNames)
	writer.writeheader()
	
	for item in data:
		writer.writerow(data[item])
print"Data written to file."

data = pd.read_csv('../dataset/preprocessed_first_50000.csv').fillna('0')
ts = data.loc[:, 'timestamp']
data = data.to_dict(orient='records')
print ts[:5]

vec = DictVectorizer()
X = np.array(vec.fit_transform(data).toarray())

X = preprocessing.robust_scale(X)
print "\nDimensions of feature matrix: ", X.shape
np.savetxt('../dataset/matX_ts.txt', X)



