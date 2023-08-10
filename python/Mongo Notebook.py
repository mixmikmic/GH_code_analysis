from pymongo import MongoClient
import datetime
import numpy as np
import pandas as pd
import getpass

database = 'ada-project'
user = input('MongoDB name: ')
password = getpass.getpass('MongoDB password: ')

# Mongo Client and authentification
client = MongoClient('www.cocotte-minute.ovh', 27017)
db = client[database]
db.authenticate(user, password)
collection = db['recipes']

# Number of recipes
collection.count()

# Find an element by ID
collection.find_one({'recipeID':47564})



