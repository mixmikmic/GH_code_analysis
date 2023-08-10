##Import Crime Data 
import pandas as pd 
crime_data=pd.read_csv('../Raw_Data/7.2017_crime_data.csv')

#Pull associated Crime Code and Location
crime_data2=crime_data[['Location ','Crime Code']]
#Drop any NaN location
crime_data2 =crime_data2.dropna(axis=0, how='any')
crime_data2.head()

#Need to change data type in order to strip out Lat and Lng
crime_data2['Location ']=crime_data2['Location '].astype('str')
# Split Lat and Lng
df = crime_data2['Location '].apply(lambda x: pd.Series(x.split(',')))
#create new columns to store Lat and Lng
df.columns=['lat','lng']
df

#Add in Crime Code
crime_data3=pd.concat([df,crime_data[['Crime Code']]],axis=1)
crime_data_copy=crime_data3[:]
crime_data3=crime_data3[crime_data3.lng.str.contains("NaN") == False]
crime_data3['lat'] = crime_data3['lat'].map(lambda x: str(x)[1:])
crime_data3['lng'] = crime_data3['lng'].map(lambda x: str(x)[:-1])
crime_data3.head()

#Pull address from lat / lng
import requests
gkey="AIzaSyDuR6Ej6fNbaY-gjZRaA0t3THaJw-UNai8"

def reverse_geocode(latitude, longitude):
    # Did the geocoding request comes from a device with a
    # location sensor? Must be either true or false
    sensor = 'true'

    # Hit Google's reverse geocoder directly
    # NOTE: I *think* their terms state that you're supposed to
    # use google maps if you use their api for anything.
    base = "https://maps.googleapis.com/maps/api/geocode/json?"
    params = "latlng={lat},{lon}&sensor={sen}&key={key}".format(
        lat=latitude,
        lon=longitude,
        sen=sensor,
        key=gkey
    )
    url = "{base}{params}".format(base=base, params=params)
    #print(url)
    response = requests.get(url).json()
    #print(response)
    address = response['results'][0]['formatted_address']
    return address

crime_data3=crime_data3.reset_index()
lat=crime_data3['lat']
lng=crime_data3['lng']
code=crime_data3['Crime Code']

##Write a loop to call geocode fuction for each lat lng in the crime_data3 
list_add=[]
for j in range(len(lat)):
    dic_add={}
    dic_add['lat']=lat[j]
    dic_add['lng']=lng[j]
    dic_add['code']=code[j]
    dic_add['add']=reverse_geocode(lat[j], lng[j])
    list_add.append(dic_add)

list_address=pd.DataFrame(list_add)
list_address

list_address['add'].apply(lambda x: pd.Series(x.split(',')))

address

