import pyspark
from pyspark import SparkContext, SQLContext
sc = pyspark.SparkContext()
sqlc = SQLContext(sc)

import os
from pyspark.sql import functions as F, Window
from pyspark.sql.functions import *
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import ParamGridBuilder
import numpy as np

class ActionsData:
    
    def __init__(self,folder,file,file_repos,user,date,item):
        self._data=None
        self._data_items=None
        self.load_data(folder,file,user,date,item)
        self.load_items(folder,file_repos)
            
    def load_data(self,foldername,filename,user,date,item):
        """load interactions data of users with repositories"""
        file=os.path.join(foldername+filename)
        data=sqlc.read.json(file)
        data=data.select(col(user).alias('user_id'),col(date).alias('created_at'), col(item).alias('repo_id'))
        self._data=data
        
    def load_items(self,foldername,filename):
        """load informations about repositories"""
        file_repos=os.path.join(foldername+filename)
        data_repos=sqlc.read.json(file_repos)
        self._data_items=data_repos.select(col('id').alias('repo_id'),'name','language').distinct()
    
    def join_w_repos(self):
        """consider only interactions to repositories contained in self._data_items"""
        self._data=self._data.join(self._data_items,'repo_id','inner')
    
    
    def remove_duplicates(self):
        """remove duplicated of interactions of a user with the same repository"""
        self._data=self._data.sort('user_id','created_at',ascending=True).dropDuplicates(['user_id','repo_id'])

    def filter_actions(self,min_actions,max_actions):
        """filter out users inactive users (users who interacted with less than min_actions
        repositories) and outliers (users who interacted with more than max_actions repositories)"""
        data_with_max=self._data.groupby('user_id').agg(F.count('repo_id').alias('total_actions'))
        data_filter=data_with_max.filter((data_with_max.total_actions>min_actions)                                           & (data_with_max.total_actions<max_actions))
        
        self._data=self._data.join(data_filter.select('user_id'),'user_id','inner')

        
    def add_rating(self,rating):
        """add a column with rating value: in a class instance each interaction has the same value"""
        self._data=self._data.groupby('user_id','created_at','repo_id')                            .agg((F.count('*')*rating).alias('rating'))
        
        
    def transform(self,min_actions,max_actions,rating):
        """apply data transformations"""
        self.join_w_repos()
        self.remove_duplicates()
        self.filter_actions(min_actions,max_actions)
        self.add_rating(rating)

class SimpleRecommender:
    
    def __init__(self,data):
        
        self._data=data
        self._train=None
        self._test=None
        self._model=None
        self._predictions_train=None
        self._predictions_test=None
    
    
    def message(self,x):
        print(x)
        
    def split_train_test(self):
        self._train=self._data.filter('number_of_actions<total_actions')
        self._test=self._data.filter('number_of_actions=total_actions')
         
    def fit(self,param):
        self.split_train_test()
        als = ALS(maxIter=param['iter'],rank=param['rank'],regParam=param['reg'],userCol="user_idn",                    itemCol="repo_idn",ratingCol="rating", seed=1, coldStartStrategy='drop')
        evaluator_reg=RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")      
        model=als.fit(self._train)
        self._model=model
        self._predictions_train=model.transform(self._train)
        train_rmse=evaluator_reg.evaluate(self._predictions_train)
        self.message('Train RMSE=' + str(train_rmse))
        self._predictions_test=model.transform(self._test)
        test_rmse=evaluator_reg.evaluate(self._predictions_test)
        self.message('Test RMSE=' + str(test_rmse))
        
        

forks=ActionsData(folder='./data',file='projects_forked_2017.json',                  file_repos='projects_not_forked_2017.json',                  user='owner_id',date='created_at',item='forked_from')

forks.transform(min_actions=5, max_actions=2500,rating=1)

forks_data=forks._data

forks_data.cache()

w=(Window.partitionBy('user_id').orderBy('created_at').rowsBetween(Window.unboundedPreceding, Window.currentRow))
forks_data=forks_data.withColumn('number_of_actions',F.count('user_id').over(w))

total_actions=forks_data.groupby('user_id').agg(F.max('number_of_actions').alias('total_actions'))

forks_data=forks_data.join(total_actions,'user_id','inner')

indexer_user=StringIndexer(inputCol="user_id",outputCol="user_idn")#.setHandleInvalid('skip')
indexer_repo=StringIndexer(inputCol='repo_id',outputCol='repo_idn')
forks_data=indexer_user.fit(forks_data).transform(forks_data)
forks_data=indexer_repo.fit(forks_data).transform(forks_data)

forks_data.cache()

rec=SimpleRecommender(forks_data)

parameters={'rank':20,'iter':20,'reg':0.1}

rec.fit(param=parameters)
model_final=rec._model

model_final.save('/data/als_r20_i20_reg01_f.parquet')

sc.stop()



