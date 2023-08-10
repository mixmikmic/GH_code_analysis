# import all the neccessary packages
import pandas as pd 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
import numpy as np
import random
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neural_network import MLPClassifier
import pickle 
import pdb
from nltk.stem import WordNetLemmatizer
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.stem import PorterStemmer
import math
from nltk.metrics.scores import accuracy, precision, recall
import collections
import os, sys
import time

# Read news files of a company
def create_news_df(news_file):
    news_df = pd.read_csv(news_file, encoding = "ISO-8859-1")
    news_df['Date'] = pd.to_datetime(news_df['Date'], errors= 'coerce')
    news_df['Text'] = news_df['Body'].astype(str).str.cat(news_df['Title'].astype(str))
    del news_df['Body']
    del news_df['Title']
    news_df.fillna('', inplace=True)
    return news_df

# Read Yahoo stock of a company
def create_stock_df(stock_file):
    #  append Target into News dataframe
    stock_df = pd.read_csv(stock_file, encoding = "ISO-8859-1")
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    return stock_df

# Combine news data and stock data of a company to prepare for label creation process
def combine_final_df(news_df, stock_df):
    df= stock_df.set_index('Date').join(news_df.set_index('Date'))
    df.fillna(0, inplace = True)
    df['Target'] = np.nan
    print(df.head())
    requirement = 0.00000
    for i in range(len(df)):
        if df['Abnormal Return'].iloc[i] > requirement:
            df['Target'].iloc[i] = 1.0
        elif df['Abnormal Return'].iloc[i] <  -requirement:
            df['Target'].iloc[i] = -1.0
        else:
            df['Target'].iloc[i] = 0.0
    return df

# We conduct experiments with data of 21 companies, so we have to combine processed data of these companies together to create 1 complete training dataset
def combine_multiple_company(company_list):  
    """
    Concatenate multiple dataframe of companies together to create big lexicon
    """
    company_dfs = []

    for company in company_list:
        print(company)
        news_file_name = 'News/'+ company + '_News.csv'
        news = create_news_df(news_file_name)
        stock_file_name = 'Abnormal_Returns/' + company + '_AbnormalReturn.csv'
        stock = create_stock_df(stock_file_name)

        final = combine_final_df(news, stock)
        company_dfs.append(final)
    total = pd.concat(company_dfs, ignore_index = True)
    return total



