import requests

response = requests.get("https://www.simplyrecipes.com/?s=asparagus+garlic")

type(response)

print(response.status_code)

response.content#.decode('utf-8')

response.content.decode('utf-8')

response.content.decode('utf-8').find('Welcome to Simply Recipes')

import json

data_string = '[{"q":[2,3],"r":3.0,"s":"SS"}]'
python_data = json.loads(data_string)
print(type(python_data))
python_data

print(type(data_string),type(python_data))
print(type(python_data[0]),python_data[0])
print(type(python_data[0]['s']),python_data[0]['s'])

#Correct
json.loads('"Hello"')

#Wrong
json.loads("Hello")

address = 'KPI, Kyiv, Ukraine'
url = 'https://maps.googleapis.com/maps/api/geocode/json?address=%s'%(address)
response = requests.get(url).json()
print(type(response))

address="KPI,Kyiv,Ukraine"
url="https://maps.googleapis.com/maps/api/geocode/json?address=%s" % (address)
try:
    response = requests.get(url)
    if not response.status_code == 200:
        print("HTTP error",response.status_code)
    else:
        try:
            response_data = response.json()
        except:
            print("Response not in valid JSON format")
except:
    print("Something wemt wrong with requests.get()")
print(type(response_data))

# Let's see what URL looks like
url

response_data

response_data['results']

for thing in response_data['results'][0]:
    print(thing)

response_data['results'][0]['geometry']

response_data['results'][0]['geometry']['location']

def get_lat_lng(address):
    import requests, time
    
    url="https://maps.googleapis.com/maps/api/geocode/json?address=%s" % (address)
    
    try:
        response = requests.get(url)
        if not response.status_code == 200:
            print('HTTP error',response.status_code)
        else:
            try:
                response_data = response.json()
            except:
                print('Response not valid JSON format')
    except:
        print('Something went wrong with requests.get')
    try:
        time.sleep(1)
        lat = response_data['results'][0]['geometry']['location']['lat']
        lng = response_data['results'][0]['geometry']['location']['lng']
    except:
        print('Try another one')
    return (lat,lng)

get_lat_lng('Sarny,Ukraine')

def get_lat_lng_incompl(address):
    #python code goes here
    import requests, time
    
    url="https://maps.googleapis.com/maps/api/geocode/json?address=%s" % (address)
    try:
        response = requests.get(url)
        if not response.status_code == 200:
            print("HTTP error",response.status_code)
        else:
            try:
                response_data = response.json()
            except:
                print("Response not in valid JSON format")
    except:
        print("Something went wrong with requests.get")
    try:
        time.sleep(1)
        propos_adr = []
        for i in range(len(response_data['results'])):
            adr = response_data['results'][i]['address_components'][0]['long_name']
            lat = response_data['results'][i]['geometry']['location']['lat']
            lng = response_data['results'][i]['geometry']['location']['lng']
            propos_adr.append((adr,lat,lng))
    except:
        print("Try another one.")
    return propos_adr    

get_lat_lng_incompl('Lon')

data_string = """
<Bookstore>
   <Book ISBN="ISBN-13:978-1599620787" Price="15.23" Weight="1.5">
      <Title>New York Deco</Title>
      <Authors>
         <Author Residence="New York City">
            <First_Name>Richard</First_Name>
            <Last_Name>Berenholtz</Last_Name>
         </Author>
      </Authors>
   </Book>
   <Book ISBN="ISBN-13:978-1579128562" Price="15.80">
      <Remark>
      Five Hundred Buildings of New York and over one million other books are available for Amazon Kindle.
      </Remark>
      <Title>Five Hundred Buildings of New York</Title>
      <Authors>
         <Author Residence="Beijing">
            <First_Name>Bill</First_Name>
            <Last_Name>Harris</Last_Name>
         </Author>
         <Author Residence="New York City">
            <First_Name>Jorg</First_Name>
            <Last_Name>Brockmann</Last_Name>
         </Author>
      </Authors>
   </Book>
</Bookstore>
"""

from lxml import etree
root = etree.XML(data_string)
print(root.tag,type(root.tag))

print(etree.tostring(root,pretty_print=True).decode('utf-8'))

for element in root.iter():
    print(element)

for child in root:
    print(child)

for child in root:
    print(child.tag)

for element in root.iter('Author'):
    print(element.find('First_Name').text,element.find('Last_Name').text)

for element in root.findall('Book/Title'):
    print(element.text)

for element in root.findall('Book/Authors/Author'):
    print(element.find('First_Name').text)

root.find('Book[@Weight="1.5"]/Authors/Author/First_Name').text

root.find('Book[@Price="15.80"]/Authors/Author/Last_Name').text

books = root.findall('Book')
print(books,type(books),sep='\n')

for i in range(len(books)):
    print(root.findall('Book/Authors/Author[@Residence="New York City"]/First_Name')[i].text,
          root.findall('Book/Authors/Author[@Residence="New York City"]/Last_Name')[i].text)    



