import requests
from bs4 import BeautifulSoup
import os
import time
import re
import json
import string
from youtube_crawler import video_crawler
from lxml import etree

def song_retrieval(artist, song, path_song):

    song_url = 'https://www.hooktheory.com/theorytab/view/' + artist + '/' + song 
    response_song = requests.get(song_url)

    soup = BeautifulSoup(response_song.text, 'html.parser')
    li_list = soup.findAll("li", { "role":"presentation"})

    section_list = []
    pk_list = []
    
    ## section 
    for i in range(len(li_list)-1):
        sec = li_list[i].text.strip().lower().replace(" ", "-")
        section_list.append(sec)
        pk_list.append(soup.findAll("div", { "role":"tabpanel", "id":sec})[0].contents[0]['id'])

    ## save xml
    for idx, pk in  enumerate(pk_list):
        req_url = 'https://www.hooktheory.com/songs/getXmlByPk?pk=' + str(pk)  ##
        response_info = requests.get(req_url)
        content = response_info.text
        
        with open(os.path.join(path_song, section_list[idx] + ".xml"), "w", encoding="utf-8") as f:
            f.write(content)
        time.sleep(0.08)
    
    ## get genre
    wikiid = soup.findAll("multiselect", { "items":"genres"})[0]['wikiid']
    response_genre = requests.get('https://www.hooktheory.com/wiki/'+ str(wikiid) +'/genres')
    genre_act_list = json.loads(response_genre.text)
    genres = []
    for g in genre_act_list:
        if g['active']:
            genres.append(g['name'])   
    ## saving
    info = {'section': section_list, 'pk':pk_list, 'song_url':song_url,
            'genres': genres, 'wikiid':wikiid}

    with open(os.path.join(path_song, 'song_info.json'), "w") as f:
         json.dump(info, f)

    ## youtube
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(content, parser=parser)
    y_id = root.find('meta').find('YouTubeID').text

    video_crawler(y_id, path_song)
    
    
def get_song_list(url_artist, quite=False):
    response_tmp = requests.get(website + url_artist)
    soup = BeautifulSoup(response_tmp.text, 'html.parser')
    item_list = soup.find_all("li", { "class":"grid-item"})

    song_name_list = []
    for item in item_list:
        song_name = item.find_all("a", { "class":"a-tab-cover"})[0]['href'].split('/')[-1]
        song_name_list.append(song_name)
        if not quite:
            print('   > %s' % song_name)
    return song_name_list    
   

list_pages = []
archive_artist = dict()
sleep_time = 0.11
alphabet_list = string.ascii_lowercase[-4:-2]

website = 'https://www.hooktheory.com'
base_url = website + '/theorytab/artists/'

artist_count = 0
song_count = 0

for ch in alphabet_list:
    time.sleep(sleep_time)    
    url = base_url+ch
    response_tmp = requests.get(url)
    soup = BeautifulSoup(response_tmp.text, 'html.parser')
    page_count = 0
    
    print('==[%c]================================================='%ch)
    
    ## get artists list by pages
    url_artist_list = []
    for page in range(1,9999):
        url = 'https://www.hooktheory.com/theorytab/artists/'+ch+'?page=' + str(page)
        
        time.sleep(sleep_time) 
        response_tmp = requests.get(url)
        soup = BeautifulSoup(response_tmp.text, 'html.parser')
        item_list = soup.find_all("li", { "class":"grid-item"})
        
        if item_list:
            print(url)
            page_count += 1
        else:
            break
            
        for item in item_list:
            url_artist_list.append(item.find_all("a", { "class":"a-tab-cover"})[0]['href'])

    print('Total:', len(url_artist_list))
    
    print('----')
    
    if not page_count:
        page_count = 1  
        
    ## get song of artists
    artist_song_dict = dict()
    
    for url_artist in url_artist_list:
        artist_count+=1
        time.sleep(sleep_time)
        artist_name = url_artist.split('/')[-1]
        print(artist_name)
        song_name_list = get_song_list(url_artist)
        song_count  += len(song_name_list)
        artist_song_dict[artist_name] = song_name_list
        
    archive_artist[ch] = artist_song_dict
    list_pages.append(page_count)

print('=======================================================')
print(list_pages)
print('Artists:', artist_count)
print('Songs:', song_count)


with open('archive_artist.json', "w") as f:
     json.dump(archive_artist, f)

root_dir = 'archive'

with open('archive_artist.json', "r") as f:
     archive_artist = json.load(f)

now_count = 1
        
for ch in alphabet_list:
    path_ch = os.path.join(root_dir, ch)
    print('==[%c]================================================='%ch)
    
    if not os.path.exists(path_ch):
        os.makedirs(path_ch)
    
    for a_name in archive_artist[ch].keys():        
        for s_name in archive_artist[ch][a_name]:

            print('(%3d/%3d) %s   %s' % (now_count, song_count, a_name , s_name))
            path_song = os.path.join(path_ch, a_name, s_name)
            
            if not os.path.exists(path_song):
                os.makedirs(path_song)
            
            time.sleep(sleep_time)  
            song_retrieval(a_name, s_name, path_song)
            
            now_count+=1 
            
            

