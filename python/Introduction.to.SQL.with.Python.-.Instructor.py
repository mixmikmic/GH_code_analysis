from IPython.display import Image
Image(url='https://vladlight.files.wordpress.com/2012/12/sakilaerm.png')

import sqlite3
import pandas as pd

# Connect to the database (again, downloaded from here: https://www.dropbox.com/s/6pz7vkl5c32xujt/sakila.db?dl=0)
con = sqlite3.connect("sakila.db")

# Set SQL query as a comment
sql_query = ''' SELECT * FROM customer '''

# Use pandas to pass sql query using connection form SQLite3
df = pd.read_sql(sql_query, con)

# Show the resulting DataFrame
df

# Set function as our sql_to_pandas

def sql_to_df(sql_query):

    # Use pandas to pass sql query using connection form SQLite3
    df = pd.read_sql(sql_query, con)

    # Show the resulting DataFrame
    return df

# Select multiple columns example
query = ''' SELECT first_name,last_name
            FROM customer; '''

# Grab from first two columns
sql_to_df(query).head()

# Select multiple colums example
query = '''SELECT * 
           FROM customer;'''

# Grab
sql_to_df(query).head()

# Select distinct country_ids from the city table.

query = '''SELECT DISTINCT(country_id)
           FROM city'''

sql_to_df(query).head()

# Select all customer info from the 1st store.

query = '''SELECT *
           FROM customer
           WHERE store_id = 1'''

sql_to_df(query).head()

query = '''SELECT *
           FROM customer
           WHERE first_name = 'MARY' '''

sql_to_df(query).head()

# Select all films from 2006 that are rated R.

query = '''SELECT *
           FROM film
           WHERE release_year = 2006
           AND rating = 'R' '''

sql_to_df(query).head()

# Select all films rated R or PG

query = '''SELECT *
           FROM film
           WHERE rating = 'PG'
           OR rating = 'R' '''

sql_to_df(query).head()

# Count the number of customers
query = ''' SELECT COUNT(customer_id)
            FROM customer; '''

# Grab 
sql_to_df(query).head()

# First the % wildcard

# Select any customers whose name start with an M
query = ''' SELECT *
            FROM customer
            WHERE first_name LIKE 'M%' ; '''

# Grab 
sql_to_df(query).head()

# Next the _ wildcard

# Select any customers whose last name ends with ing
query = ''' SELECT *
            FROM customer
            WHERE last_name LIKE '_ING' ; '''

# Grab 
sql_to_df(query).head()

# Finally the [character_list] wildcard

# Select any customers whose first name begins with an A or a B
query = ''' SELECT *
            FROM customer
            WHERE first_name GLOB '[AB]*' ; '''

# Grab 
sql_to_df(query).head()

# Select all customers and order results by last name
query = ''' SELECT *
            FROM customer
            ORDER BY last_name ; '''

# Grab 
sql_to_df(query).head()

# Select all customers and order results by last name, DESCENDING
query = ''' SELECT *
            FROM customer
            ORDER BY last_name DESC; '''

# Grab 
sql_to_df(query).head()

# Count the number of customers per store

query = ''' SELECT store_id , COUNT(customer_id)
            FROM customer
            GROUP BY store_id; '''

# Grab 
sql_to_df(query).head()

