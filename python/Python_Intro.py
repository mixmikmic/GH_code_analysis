a = 3 #integer a = 3
b = "Hello" #string 
a = str(a) #cast a, an int, as a string
print(type(a))

mylist = [1,2,3,4,5]
print(mylist)
print(len(mylist))
print(mylist*2) #double the list 

a = "a" + "b"
print(a)
a = 3 + 5 #you can change the type by simply reassigning a variable 
print(a)

print( [x**2 for x in mylist]) #list comprehension does something to each element of a list (squaring)

for x in mylist: #for loop
    print (x**2) #print squared value at each iteration

anotherlist = ["a","b","c","d","e"] #another list, now containing strings
mydict = {} #create an empty dictionary
# A dictionary has key:value pairs 
for i in range(0,len(anotherlist)): #loop through each element of anotherlist
    mydict[anotherlist[i]] = mylist[i] #assign new dictionary key:value pairs
print(mydict)

my_dict = {"Gerry":2, "Jillian":1} #assign values directly to a dictionary
print(my_dict)
my_dict["Gerry"]
my_dict["John"] = 3 #add a new key:value to my_dict
print(my_dict)
my_dict1 = {3:4, 5:6}

#dictionary new_dict = {}
#list new_list = []
new_list = []
new_dict = {}
new_dict["Gerry"] = 2
new_dict["John"] = 4
new_list.append(2)
new_list.append(3)
new_list.append(4)
print(new_list)
print(new_dict)

new_dict["Gerry"] #return the value corresponding to the key "Gerry"

x = list(new_dict.keys())
for i in x:
    print(i)

#web scraping
#inport packages
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

#search_url string
search_url = 'http://www.bbc.com/'
#open with Beautiful Soup and collect HTML tags
soup = BeautifulSoup(urlopen(search_url).read(), 'html.parser')
#find classes containing headlines
headlines = soup.find_all("a", class_="block-link__overlay-link")

headline_list = []
for h in headlines:
    #convert from BeautifulSoup object to text
    #remove trailing and leading whitespace with str.strip()
    headline = str(h.get_text())
    headline_list.append(headline.strip())
    
#for h in headline_list:
#    print(h)
    
s = "a b c d e"
#print(len(s.split(" ")))

headline_length = {}
for headline in headline_list:
    headline_length[headline] = len(headline.split(" "))

#print(headline_length)
#turn headline_list into Pandas series object
series = pd.Series(headline_list)
#write to CSV --> will save to same folder that jupyter is opened from
series.to_csv("headline_list.csv")

#write the dictionary to csv
df = pd.DataFrame.from_dict(headline_length, orient="index", dtype=None)
df.to_csv("dictionary_csv.csv")

#read the csv
csv_df = pd.read_csv("dictionary_csv.csv")
#rename the columns
#test by opening in Excel through your finder/Windows explorer
csv_df.columns = ['Headline', 'Length']
print(csv_df)


