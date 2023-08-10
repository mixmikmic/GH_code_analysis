from bs4 import BeautifulSoup
import requests
import re
import time
import timeit
import random

# spoof Firefox request
req_headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:43.0) Gecko/20100101 Firefox/43.0'
}

req = requests.get('http://www.azlyrics.com/m/muse.html', headers=req_headers)
main_lyrics_page = req.text

# find main lyrics div to get a list of songs to scrape
soup = BeautifulSoup(main_lyrics_page, 'html.parser')
main_div = soup.select('div#listAlbum')
songs = []
if len(main_div) > 0:
    # there should only be one item that matches the selection
    lyrics = main_div[0]
    # find all but the first row (as the first is the header)
    for m in lyrics.findAll('a'):
        if('href' in m.attrs and 'lyrics/' in m.attrs['href']):
            songs.append({ 'title': m.text, 'link': m.attrs['href'] })    

t1 = timeit.default_timer()

n_songs = len(songs)
# now loop through dict, fetch lyrics from page and store back in dict
for s in songs[0:n_songs]:
    print('Fetching lyrics for {}...'.format(s['title']))
    s_link = 'http://azlyrics.com' + s['link'][2:]
    lreq = requests.get(s_link, headers=req_headers)
    s_raw = lreq.text
    l_soup = BeautifulSoup(s_raw, 'html.parser')
    s_div = l_soup.select('div.ringtone')
    frm = s_div[0]
    f = frm.find_next_siblings()[1]
    s_lyrics_raw = f.contents[1].text.replace('\r', '')
    # extract until the 'Submit Corrections' bit,
    # and replace multiple new lines with single new line
    s_lyrics = re.sub('\n+', '\n', s_lyrics_raw[:s_lyrics_raw.find('Submit Corrections')])
    # get rid of weird apostrophe characters
    s_lyrics = s_lyrics.replace('â\x80\x99', '\'')
    # set the lyrics back in the dict
    s['lyrics'] = s_lyrics
    # wait a bit before scraping the next page to avoid spamming
    time.sleep(2)
t2 = timeit.default_timer()
print('---------------------------------------\nFinished scraping {} songs in {} seconds'.format(n_songs, round(t2-t1, 1)))

songs_with_lyrics = []
for s in songs[0:n_songs]:
    if 'lyrics' in s:
        songs_with_lyrics.append(s)
len(songs_with_lyrics)

final_text = ''
for s in songs_with_lyrics:
    final_text = final_text + s['lyrics']

with open('lyrics.txt', 'w') as f:
    f.write(final_text)

import re
import random

class Markov(object):
    def __init__(self, raw_text):
        # extract words
        self.word_list = self.extract_list(raw_text)
        # extract triplets
        self.words = self.generate_triples(self.word_list)
        
    def extract_list(self, raw_text):
        # find all lines of text or new lines
        text_only = re.sub(r'[^.a-zA-Z0-9 \n]', '', raw_text)
        m = re.findall('([a-z| ]+|\n)', text_only, re.MULTILINE | re.IGNORECASE)
        # m is a list of entire rows and new lines
        # use nested list comprehension to split each line into words, and concatenate with new lines
        return [item.lower() for sublist in [x.split(' ') for x in m] for item in sublist]
    
    def generate_triples(self, word_list):
        d = {}
        # loop through the text and generate triples
        for i in range(len(word_list)):
            if i == 0 or i == len(word_list) - 1:
                continue
            else:
                if (word_list[i-1], word_list[i]) in d:
                    d[(word_list[i-1], word_list[i])].append(word_list[i+1])
                else:
                    d[(word_list[i-1], word_list[i])] = []
                    d[(word_list[i-1], word_list[i])].append(word_list[i+1])
        return d
    
    def get_random_pair(self):
        pair = random.choice(list(self.words))
        return [pair[0], pair[1]]
    
    def get_random_word(self, phrase=None):
        if phrase:
            # find a phrase from the list of words associated with the last two words in the supplied phrase
            phrase_words = [x.lower() for x in phrase.split(' ')]
            if len(phrase_words) > 1:
                if (phrase_words[-2], phrase_words[-1]) in self.words:
                    past = self.words[(phrase_words[-2], phrase_words[-1])]
                else:
                    past = self.words[random.choice(list(self.words.keys()))]
            else:
                past = self.words[random.choice(list(self.words.keys()))]
        else:
            # no phrase supplied, return a word from our dict at random
            past = self.words[random.choice(list(self.words.keys()))]
        return random.choice(past)
    
    def generate_song(self, start_phrase=None):
        song = self.get_random_pair()
        for _ in range(1,3):
            song.extend([self.get_random_word(' '.join([song[-2], song[-1]]))])
        song.extend(['\n\nVerse 1\n\n'])
        # if a starting phrase was supplied
        if start_phrase:
            # extract individual words
            start_words = start_phrase.split(' ')
            # if two or more words were supplied
            if len(start_words) > 1:
                # the generated text will start with the starting phrase
                song.extend(start_words)
            else:
                song.extend(self.get_random_pair())
        else:
            song.extend(self.get_random_pair())
        
        # generate a verse
        for _ in range(random.randint(15,25)):
            song.extend([self.get_random_word(' '.join([song[-2], song[-1]]))])
        
        # generate a chorus
        chorus = ['\n\nChorus\n\n']
        chorus.extend(self.get_random_pair())
        for _ in range(random.randint(12,20)):
            chorus.extend([self.get_random_word(' '.join([chorus[-2], chorus[-1]]))])
        
        # add the chorus to the song
        song.extend(chorus)
        
        # generate another verse
        song.extend(['\n\nVerse 2\n\n'])
        for _ in range(random.randint(15,25)):
            song.extend([self.get_random_word(' '.join([song[-2], song[-1]]))])
        
        # add the chorus to the song again
        song.extend(chorus)
        
        # return the song as a string
        return ' '.join(song)

with open('lyrics.txt', 'r') as f:
    s = ''.join(f.readlines())
m = Markov(s)

print(m.generate_song('you are just'))

