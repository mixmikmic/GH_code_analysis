from bs4 import BeautifulSoup
import requests

#Download the page and convert it into a beautiful soup object
yahoo_sports_url = "https://sports.yahoo.com/nba/stats/"
res = requests.get(yahoo_sports_url)
soup = BeautifulSoup(res.content, 'html.parser')

#Get urls in page via the 'a' tag and filter for nba/players in urls
nba_player_urls = []
for link in soup.find_all('a'):
    link_url = link.get('href')
    #Discard "None"
    if link_url:
        if "nba/players" in link_url:
            print(link_url)
            nba_player_urls.append(link_url)

#look at a single link
one_url = nba_player_urls[0]
res_one_url = requests.get(one_url)
soup_one_url = BeautifulSoup(res_one_url.content, 'html.parser')

#find the line with Birth Place
lines = soup_one_url.text
res2 = lines.split(",")
key_line = []
for line in res2:
    if "Birth" in line:
        print(line)
        key_line.append(line)

# Extract Birthplace
birth_place = key_line[0].split(":")[-1].strip()
print(birth_place)

def find_birthplaces(urls):
    """Get the Birthplaces"""
    for url in urls:
        profile = requests.get(url)
        profile_url = BeautifulSoup(profile.content, 'html.parser')
        lines = profile_url.text
        res2 = lines.split(",")
        key_line = []
        for line in res2:
            if "Birth" in line:
                #print(line)
                key_line.append(line)
        birth_place = None
        try:
            birth_place = key_line[0].split("*")[-1].strip()
        except IndexError:
            print(f"skipping {url}")
        print(birth_place)

        
find_birthplaces(nba_player_urls)



