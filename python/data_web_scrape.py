import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

url = 'http://www.xeno-canto.org/species/Cercomacra-tyrannina?pg=1&dir=0&order=typ'
soup = BeautifulSoup(requests.get(url).text, 'lxml')

soup_table = soup.find(class_="results")

icons = soup_table.find_all('td')

print icons[11]

print icons[11].find('a')['href']

# got it working - now to make it fully automated for song only

# using the following birds based on previous research and geographical interest
names = ['dusky_antbird', 'great_antshrike', 'barred_antshrike', 'purple_finch', 'northern_cardinal', 
         'black-capped_chickadee', 'american_goldfinch', 'tufted_titmouse', 'american_robin', 'baltimore_oriole', 
         'bluejay', 'bluebird', 'house_finch']

urls = ['http://www.xeno-canto.org/species/Cercomacra-tyrannina?pg=1', 
        'http://www.xeno-canto.org/species/Taraba-major?pg=1', 
        'http://www.xeno-canto.org/species/Thamnophilus-doliatus?pg=1', 
        'http://www.xeno-canto.org/species/Haemorhous-purpureus?pg=1', 
        'http://www.xeno-canto.org/species/Cardinalis-cardinalis?pg=1', 
        'http://www.xeno-canto.org/species/Poecile-atricapillus?pg=1', 
        'http://www.xeno-canto.org/species/Spinus-tristis?pg=1', 
        'http://www.xeno-canto.org/species/Baeolophus-bicolor?pg=1', 
        'http://www.xeno-canto.org/species/Turdus-migratorius?pg=1', 
        'http://www.xeno-canto.org/species/Icterus-galbula?pg=1', 
        'http://www.xeno-canto.org/species/Cyanocitta-cristata?pg=1', 
        'http://www.xeno-canto.org/species/Sialia-sialis?pg=1', 
        'http://www.xeno-canto.org/species/Haemorhous-mexicanus?pg=1'
       ]

all_urls = []

for url in urls:
    soup = BeautifulSoup(requests.get(url).text, 'lxml')
    pages = soup.find(class_='results-pages')
    last_page = int(pages.find_all('li')[-2].text)
    all_urls.append([url[:-1]+str(i) for i in range(1,last_page+1)])

# check we have correct urls

all_urls[1]

# get download links

download_links = []
for species in all_urls:
    download_links.append([])
    for page in species:
        soup = BeautifulSoup(requests.get(page).text, 'lxml')
        soup_table = soup.find(class_="results")
        soup_rows = soup_table.find_all('tr')
        for soup_row in soup_rows[1:]:
            elements = soup_row.find_all('td')
            if elements[9].text.lower() == 'song ':
                download_links[-1].append('http://www.xeno-canto.org'+elements[11].find('a')['href'])   

# write links to to file

for species in download_links:
    with open('bird_links/'+names[download_links.index(species)]+'.txt', 'w') as f:
        for link in species:
            f.write(link+'\n')

# now modify so that we find only those with 'A' rating

download_links = []
for species in all_urls:
    download_links.append([])
    for page in species:
        soup = BeautifulSoup(requests.get(page).text, 'lxml')
        soup_table = soup.find(class_="results")
        soup_rows = soup_table.find_all('tr')
        for soup_row in soup_rows[1:]:
            elements = soup_row.find_all('td')
            if elements[11].find(class_='selected') == None:
                continue
            if elements[9].text.lower() == 'song ' and elements[11].find(class_='selected').text == 'A':
                download_links[-1].append('http://www.xeno-canto.org'+elements[11].find('a')['href'])

# check how many recordings we have for each bird

for i in range(len(download_links)):
    print names[i], len(download_links[i])

# condense so we only have the top 6 birds
# write links to file so we can wget to download

zipped = zip(names, download_links)
cond_download_links = sorted(zipped, key=lambda x: len(x[1]), reverse=True)[:6]

for species in cond_download_links:
    with open('bird_links/A-bird_links/'+species[0]+'.txt', 'w') as f:
        for link in species[1]:
            f.write(link+'\n')

# condense to only take 6 birds with top number of 'A' recordings 
# use wget to download the mp3 files

for species in cond_download_links:
    # this will use the wget bash command to download the links in the textfile and save it to the location specified
    # !wget --content-disposition 'bird_links/A-bird_links/'+species[0]+'.txt' -P 'bird_audio/A_rate_only/mp3/A-bird'+str(birdnum)
    print 'Finished downloading ' + species[0]

# take input from the 'A' mp3 file directories and convert to wav files

from pydub import AudioSegment
import os

for i in os.listdir(os.path.abspath('bird_audio/A_rate_only/mp3/')):
    if i[0] == 'A':
        for song in os.listdir(os.path.abspath('bird_audio/A_rate_only/mp3/'+i)):
            if song[0] == '.':
                continue
            sound = AudioSegment.from_mp3(os.path.abspath('bird_audio/A_rate_only/mp3/'+i+'/'+song))
            sound.export(os.path.abspath('bird_audio/A_rate_only/wav/wav_'+i+'/'+song[:-4]+'.wav'), format="wav")



