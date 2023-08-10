#Original data source
#http:§§//www.content.digital.nhs.uk/catalogue/PUB23139

#Get the datafile
get_ipython().system('wget -P data http://www.content.digital.nhs.uk/catalogue/PUB23139/gp-reg-patients-LSOA-alt-tall.csv')

#Import best ever data handling package
import pandas as pd

#Load downloaded CSV file
df=pd.read_csv('data/gp-reg-patients-LSOA-alt-tall.csv')
#Preview first few lines
df.head()

import sqlite3

#Use homebrew database of NHS administrative info
con = sqlite3.connect("nhsadmin.sqlite")

ccode='10L'

#Find 
EPRACCUR='epraccur'
epraccur_iw = pd.read_sql_query('SELECT * FROM {typ} WHERE "Commissioner"="{ccode}"'.format(typ=EPRACCUR,ccode=ccode), con)

epraccur_iw

import folium
#color brewer palettes: ‘BuGn’, ‘BuPu’, ‘GnBu’, ‘OrRd’, ‘PuBu’, ‘PuBuGn’, ‘PuRd’, ‘RdPu’, ‘YlGn’, ‘YlGnBu’, ‘YlOrBr’, and ‘YlOrRd’.

#Fiona is a powerful library for geo wrangling with various dependencies that can make installation a pain...
#...but I have it installed already so I can use it to trivially find the centre of a set of boundaries in a geojson file
import fiona

#This is a canned demo - I happen to have the Local Authority Code for the Isle of Wight...
#...and copies of LSOA geojson files by LA
# (I could get LA code from the NHS addmin db)
geojson_local='../../IWgeodata/lsoa_by_lad/E06000046.json'
fi=fiona.open(geojson_local)
centre_lat,centre_lon=((fi.bounds[0]+fi.bounds[2])/2,(fi.bounds[1]+fi.bounds[3])/2)

#Add a widget in that lets you select the GP practice by name then fudge the lookup to practice code
#We could also add another widget to select eg Male | Female | All
def generate_map(gpcode):
    gpmap = folium.Map([centre_lon,centre_lat], zoom_start=11)
    gpmap.choropleth(
        geo_path=geojson_local,
        data=df[df['PRACTICE_CODE']==gpcode],
        columns=['LSOA_CODE', 'ALL_PATIENTS'],
        key_on='feature.properties.LSOA11CD',
        fill_color='PuBuGn', fill_opacity=0.7,
        legend_name='Number of people on list in LSOA'
        )
    
    return gpmap

def generate_map_from_gpname(gpname):
    gpcode=epraccur_iw[epraccur_iw['Name']==gpname]['Organisation Code'].iloc[0]
    return generate_map(gpcode)

#iw_gps=epraccur_iw['Organisation Code'].unique().tolist()
iw_gps=epraccur_iw['Name'].unique().tolist()
iw_gps[:3],len(iw_gps)

from ipywidgets import interact
interact(generate_map_from_gpname, gpname=iw_gps);





