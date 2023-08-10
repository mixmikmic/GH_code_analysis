import sqlite3
import datetime as dt   
import time
import csv
import requests
import pandas as pd, numpy as np
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('pylab inline')

# helper functions

def sqlite_test():
	try:
		conn = sqlite3.connect('/Users/GGV/Desktop/mcdonald_listings.sqlite')
		c = conn.cursor()
		query = "select * from trafi limit 10 ;"
		df = pd.read_sql_query(query, conn)
		print (df)
		print ("Sqlite connect OK")
	except:
		print ("Sqlite connect failed")




def sqlite_query_df(query):
	try:
		conn = sqlite3.connect('/Users/GGV/Desktop/mcdonald_listings.sqlite')
		c = conn.cursor()
		df = pd.read_sql_query(query, conn)
		print ("Sqlite connect OK")
		return df 
	except:
		print ("Sqlite connect failed")

query= '''

select * 
FROM listings 
'''
listings = sqlite_query_df(query)

query= '''

select * 
FROM reviews 
'''
reviews = sqlite_query_df(query)

reviews.head(2)

listings.head(2)

query_avg_like= '''

SELECT avg(likes)
FROM listings

'''
avg_like = sqlite_query_df(query_avg_like)
avg_like

query_avg_like_used= '''

SELECT avg(likes)
FROM listings
WHERE condition == 'Used'
'''
avg_like_used = sqlite_query_df(query_avg_like_used)
avg_like_used

query_avg_price_m_keyword= '''

SELECT avg(price)
FROM listings
WHERE LOWER(title) LIKE '%%mcdonalds%%'

'''
avg_price_m_keyword = sqlite_query_df(query_avg_price_m_keyword)
avg_price_m_keyword

query_num_unrelated_m= '''


WITH LIST AS
  ( SELECT *
   FROM listings
   WHERE LOWER(title) NOT LIKE '%%mcdonald%%'
     AND LOWER(description) NOT LIKE '%%mcdonald%%' )
SELECT COUNT(*)
FROM LIST


'''
num_unrelated_m = sqlite_query_df(query_num_unrelated_m)
num_unrelated_m

query_top_3_seller= '''

SELECT *
FROM reviews
ORDER BY positive DESC LIMIT 3

'''
top_3_seller = sqlite_query_df(query_top_3_seller)
top_3_seller

query_review= '''
SELECT count(*)
FROM
  (SELECT username,
          sum(negative) + sum(positive) + sum(neutral) AS sum_
   FROM reviews
   GROUP BY 1
   HAVING sum_ > 200
   ORDER BY sum_ DESC) sub

'''
review = sqlite_query_df(query_review)
review

query_only_positive_review= '''

SELECT count(*)
FROM
  (SELECT username,
          sum(negative),
          sum(positive),
          sum(neutral)
   FROM reviews
   GROUP BY 1
   HAVING sum(negative) = 0
   AND sum(neutral) = 0
   AND sum(positive) > 0 ) sub

'''
positive_review = sqlite_query_df(query_only_positive_review)
positive_review

query_list_positive_review= '''

WITH user_list AS
  (SELECT username
   FROM reviews
   GROUP BY 1
   HAVING sum(negative) = 0
   AND sum(neutral) = 0
   AND sum(positive) > 0)
SELECT count(*)
FROM listings
JOIN user_list ON listings.username = user_list.username
ORDER BY listings.username
  

'''
list_positive_review = sqlite_query_df(query_list_positive_review)
list_positive_review

query_top_user_detail= '''

SELECT username,
       count(username),
       sum(price),
       sum(likes)
FROM listings
GROUP BY username
ORDER BY count(username) DESC LIMIT 3

'''
top_user_detail = sqlite_query_df(query_top_user_detail)
top_user_detail

query_top_user_list_review = '''

WITH top_user AS
  (SELECT username,
          sum(negative) + sum(positive) + sum(neutral) AS sum_
   FROM reviews
   GROUP BY 1
   ORDER BY sum_ DESC LIMIT 3),
     listing AS
  ( SELECT top_user.username AS username,
           count(listings.username) AS listing_
   FROM listings
   JOIN top_user ON top_user.username = listings.username
   GROUP BY 1 )
SELECT top_user.*,
       listing.listing_
FROM listing
JOIN top_user ON top_user.username = listing.username

'''
top_user_list_review = sqlite_query_df(query_top_user_list_review)
top_user_list_review

