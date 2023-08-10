# The code was removed by DSX for sharing.

import json
from watson_developer_cloud import NaturalLanguageClassifierV1

natural_language_classifier = NaturalLanguageClassifierV1(
    username=credentials_1['username'],
    password=credentials_1['password'])

classifiers = natural_language_classifier.list()
print(json.dumps(classifiers, indent=2))

#create a classifier
import urllib
urllib.urlretrieve ("https://raw.githubusercontent.com/analytics-bootcamp/Training-material/master/7.%20Training%20-%20Misc/weather.txt", "weather.txt")

with open('weather.txt', 'rb') as training_data:
     print(json.dumps(natural_language_classifier.create(training_data=training_data, name='weather'), indent=2))

# replace 2374f9x68-nlc-2697 with your classifier id
status = natural_language_classifier.status('4d5c10x177-nlc-2873')
print(json.dumps(status, indent=2))

status = natural_language_classifier.status('4d5c10x177-nlc-2873')
print(json.dumps(status, indent=2))

if status['status'] == 'Available':
    classes = natural_language_classifier.classify('4d5c10x177-nlc-2873','How hot will it be tomorrow?')
    print(json.dumps(classes, indent=2))


delete = natural_language_classifier.remove('4d5c10x177-nlc-2873')
print(json.dumps(delete, indent=2))

