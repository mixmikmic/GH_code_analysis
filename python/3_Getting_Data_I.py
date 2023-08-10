import numpy as np
import pandas as pd

# reading companies file: throws an error
# companies = pd.read_csv("companies.txt", sep="\t")

# Using encoding = "ISO-8859-1"
companies = pd.read_csv("companies.txt", sep="\t", encoding = "ISO-8859-1")
companies.head()

import pymysql

# create a connection object 'conn'
conn = pymysql.connect(host="localhost", # your host, localhost for your local machine
                     user="root", # your username, usually "root" for localhost
                      passwd="yourpassword", # your password
                      db="world") # name of the data base; world comes inbuilt with mysql

# create a cursor object c
c = conn.cursor()

# execute a query using c.execute
c.execute("select * from city;")

# getting the first row of data as a tuple
all_rows = c.fetchall()

# to get only the first row, use c.fetchone() instead

# notice that it returns a tuple of tuples: each row is a tuple
print(type(all_rows))

# printing the first few rows
print(all_rows[:5])

df = pd.DataFrame(list(all_rows), columns=["ID", "Name", "Country", "District", "Population"])
df.head()

import requests, bs4

# getting HTML from the Google Play web page
url = "https://play.google.com/store/apps/details?id=com.facebook.orca&hl=en"
req = requests.get(url)

# create a bs4 object
# To avoid warnings, provide "html5lib" explicitly
soup = bs4.BeautifulSoup(req.text, "html5lib")

# getting all the text inside class = "review-body"
reviews = soup.select('.review-body')
print(type(reviews))
print(len(reviews))
print("\n")

# printing an element of the reviews list
print(reviews[6])

# step 1: split using </span>
r = reviews[6]
print(type(r))

# r is a bs4 tag, convert it into string to use str.split() 
part1 = str(r).split("</span>")
print(part1)

# Now split part1 using '<div class="review-link' as the separator
# and get the first element of the resulting list
# we get the review
part2 = part1[1].split('<div class="review-link')[0]
part2

# apply this sequence of splitting to every element of the reviews list
# using a list comprehension
reviews_text = [str(r).split("</span>")[1].split('<div class="review-link')[0] for r in reviews]

# printing the first 10 reviews
reviews_text[0:10]

# Practice problem: Scraping amazon.in to get shoe price data 
import pprint

url = "https://www.amazon.in/s/ref=nb_sb_noss?url=search-alias%3Daps&field-keywords=sport+shoes"
req = requests.get(url)

# create a bs4 object
# To avoid warnings, provide "html5lib" explicitly
soup = bs4.BeautifulSoup(req.text, "html5lib")

# get shoe names
# shoe_data = soup.select('.a-size-medium')
# shoe_data = soup.select('.a-size-small.a-link-normal.a-text-normal')
# print(len(shoe_data))
# print(pprint.pprint(shoe_data))

# get shoe prices
shoe_prices = soup.select('.a-price-whole')
print(len(shoe_prices))
pprint.pprint(shoe_prices)

