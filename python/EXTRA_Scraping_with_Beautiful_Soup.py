import requests
import bs4

# If you don't have Beautiful Soup, install with 'conda install beautifulsoup'

response = requests.get('http://duspviz.mit.edu/_assets/data/sample.html')
print(response.text) # Print the output

soup = bs4.BeautifulSoup(response.text, "html.parser")
print(soup.prettify()) # Print the output using the 'prettify' function

# Access the title element
soup.title

# Access the content of the title element
soup.title.string

# Access data in the first 'p' tag
soup.p

# Access data in the first 'a' tag
soup.a

# Retrieve all links in the document (note it returns an array)
soup.find_all('a')

# Retrieve elements by class equal to link using the attributes argument
soup.findAll(attrs={'class' : 'link'})

# Retrieve a specific link by ID
soup.find(id="link3")

# Access Data in the table (note it returns an array)
soup.find_all('td')

data = soup.findAll(attrs={'class':'city'})
print(data[0].string)
print(data[1].string)
print(data[2].string)
print(data[3].string)

data = soup.findAll(attrs={'class':'city'})
for i in data:
    print(i.string)

data = soup.findAll(attrs={'class':['city','number']})
print(data)

print(data[0])
print(data[1])

print(data[0].string)
print(data[1].string)

import requests
import bs4

# load and get the website
response = requests.get('http://duspviz.mit.edu/_assets/data/sample.html')

# create the soup
soup = bs4.BeautifulSoup(response.text, "html.parser")

# find all the tags with class city or number
data = soup.findAll(attrs={'class':['city','number']})

# print 'data' to console
print(data)

f = open('rat_data.txt','a') # open new file, make sure path to your data file is correct

p = 0 # initial place in array
l = len(data)-1 # length of array minus one

f.write("City, Number\n") #write headers

while p < l: # while place is less than length
    f.write(data[p].string + ", ") # write city and add comma
    p = p + 1 # increment
    f.write(data[p].string + "\n") # write number and line break
    p = p + 1 # increment

f.close() # close file



