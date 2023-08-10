from bs4 import BeautifulSoup
import urllib2 
import re 
import pandas as pd 
from IPython.core.display import display, HTML 
from wordcloud import WordCloud # to plot wordclouds 
import matplotlib.pyplot as plt 

plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')

prefix = 'http://lyrics.wikia.com'

def plot_word_cloud(corpus, max_words = 42, width=600, height=400, fig_size=(8,6)):
    try:
        if len(corpus) == 0:
            corpus = 'no words'
        wordcloud = WordCloud(max_words = max_words, width=width, height=height, background_color="black").generate(corpus)
        plt.figure(figsize=fig_size, dpi=80)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        return
    except:
        pass
    return
    
def get_lyrics(band_name = None
               , display_cover=False
               , show_song_word_cloud=False
               , show_album_word_cloud=False
               , verbose=False):
    """
    Asks for the artists name and download all its lyrics.
    """
    def get_artist_link():
        """
        Asks for a term to search and returns the first result.
        """
        url_search = 'http://lyrics.wikia.com/wiki/Special:Search?query='
        
        if band_name == None:
            site = urllib2.urlopen(url_search + raw_input("Artist's name ?: ").replace(' ', '+')).read()
        else:
            site = urllib2.urlopen(url_search + band_name.replace(' ', '+')).read()
        soup = BeautifulSoup(site)

        links = []
        for link in soup.find_all("a", class_='result-link'):
            if link.get('href') <> None:
                links.append(link.get('href'))
                
        print 'Getting lyrics from...', links[0]
        return links[0]
    
    def display_thumbnail(soup):
        images = soup.find_all("img", class_='thumbborder ')
        for image in images:
            display(HTML(str(image)))
        return
    
    def get_album_links(artist_link):
        link_discs = []
        site = urllib2.urlopen(artist_link).read()
        soup = BeautifulSoup(site)

        discs = soup.find_all("span", class_="mw-headline")
        for d in discs:
            for element in d.find_all('a'):
                link_discs.append(prefix + element.get('href'))

        return link_discs
    
    def get_text(lyric):
        text = ''
        for line in lyric:
            text += line
            
        print camel_case_split(text)
        return camel_case_split(text)
    
    def get_lyrics(url):
        try:
            site = urllib2.urlopen(url).read()
            soup = BeautifulSoup(site)
            lyric = soup.find_all("div", class_="lyricbox")

            if len(lyric) > 0:
                for element in lyric:
                    return re.sub("([a-z])([A-Z])","\g<1> \g<2>", BeautifulSoup(str(element).replace('<br/>',' ')).get_text())
        except:
            pass
        
    def get_list_of_links(url, link_filter):
        links = []
        site = urllib2.urlopen(url).read()
        soup = BeautifulSoup(site)
        
        if (display_cover):
            display_thumbnail(soup)   # displays the albums image

        for link in soup.find_all("a"):
            if link.get('href') <> None and '/wiki/' + link_filter + ':' in link.get('href') and not '?' in link.get('href'):
                links.append(prefix + link.get('href'))

        return links
    
    def download_lyrics(album_links):
        lyrics = []  # list with all the lrrics
        songs = []   # list with scanned links
        discography = []
        i = 1
        
        for album_link in album_links:
            album = []
            print 'Downloading:', i, 'out of', len(album_links), 'albums -', album_link.split(':')[-1].replace('_', ' ')
            i+=1
            for link in get_list_of_links(album_link, link_filter):
                if get_lyrics(link) <> None  and link not in songs:
                    lyrics.append(get_lyrics(link))
                    lyric = get_lyrics(link)
                    
                    album.append(lyric)
                    
                    if verbose:
                        print link.split(':')[-1].replace('_',' ') #print song title
                    if (show_song_word_cloud):
                        plot_word_cloud(lyric.lower(), max_words=50, width=400, height=200)
                    songs.append(link)

            if show_album_word_cloud:
                plot_word_cloud(str(album[:]).lower(), max_words=50, width=800, height=500)
            discography.append((album_link.split(':')[-1].replace('_', ' '), album))

        print '\nDone!', len(songs), 'lyrics aquired from', len(album_links), 'albums.'
        
        return discography
    
    
        
    artist_link = get_artist_link()
    link_filter = artist_link.split('/')[-1]
    album_links = get_album_links(artist_link)
    lyrics = download_lyrics(album_links)
    
    return lyrics

corpus = get_lyrics(display_cover = False # displays the cover of the album while is been proceesed
                    #, band_name='metallica' # name of the artist
                    , verbose = False # print the song titles
                    , show_album_word_cloud = False # shows a word-cloud per album
                    , show_song_word_cloud = False) # shows a word-cloud per song

# raw will contains all the text of the lyrics.
raw = ''
for title, songs in corpus:
    for song in songs:
        raw+=song

from IPython.core.display import display, HTML

html_code = """
<form action="http://www.wordle.net/advanced" method="POST" target="_blank">
    <textarea name="text" style="display:none">
        %s
    </textarea>
    <input type="submit" value="Get Wordle">
</form>""" % (raw.lower())

display(HTML(html_code))

import nltk
from __future__ import division
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords # Filter out stopwords, such as 'the', 'or', 'and'

# A plot of how many words per album chronologically and without repetition of the songs.
df1 = pd.DataFrame(columns=('album', 'songs', 'words'))
i = 0
for album_title, songs in corpus:
    words = 0
    for song in songs:
        #print len(song)
        words += len(song)
    df1.loc[i] = (album_title, len(songs), words)
    i += 1

df1.plot.bar(x='album', y='words', title='Words per Album');

tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(raw.lower())   # tokens without punctuation
text = nltk.Text(tokens)

words = [w.lower() for w in tokens]
vocab = sorted(set(words))

filtered_words = [word for word in words if word not in stopwords.words('english') and len(word) > 1 and word not in ['na','la']] # remove the stopwords
fdist = nltk.FreqDist(filtered_words)

def plot_freq_words(fdist):
    df = pd.DataFrame(columns=('word', 'freq'))
    i = 0
    for word, frequency in fdist.most_common(21):
        df.loc[i] = (word, frequency)
        i += 1

    title = 'Top %s words in lyrics' % top_n
    df.plot.barh(x='word', y='freq', title=title, figsize=(5,5)).invert_yaxis()
    
    return
    
top_n = 20
text.dispersion_plot([str(w) for w, f in fdist.most_common(top_n)])
plot_freq_words(fdist)
plot_word_cloud(raw.lower(), max_words=100, fig_size=(10,8))

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

df = pd.DataFrame(columns=('album', 'pos', 'neg', 'neu'))
df2 = pd.DataFrame(columns=('album', 'pos', 'neg', 'neu'))
i = 0
tot_pos = 0
tot_neg = 0
tot_neu = 0
for album_title, songs in corpus:
    pos = 0
    neg = 0
    neu = 0
    #print album_title
    for song in songs:
        ss = sid.polarity_scores(song)
        if ss['compound'] >= 0.5:
            pos+=1
            tot_pos+=1
        elif ss['compound'] <= 0.5:
            neg+=1
            tot_neg+=1
        else:
            neu+=1
            tot_neu+=1
    df.loc[i] = (album_title, pos, neg, neu)
    if (pos+neg+neu) > 0:
        df2.loc[i] = (album_title, pos / (pos+neg+neu), neg / (pos+neg+neu), neu / (pos+neg+neu))
    i += 1
    
df.plot.bar(x='album',stacked=True);

#print tot_pos, tot_neg, tot_neu, tot_pos / (tot_pos+tot_neg+tot_neu), tot_neg / (tot_pos+tot_neg+tot_neu), tot_neu / (tot_pos+tot_neg+tot_neu)

labels = 'Positive', 'Negative', 'Neutral'
sizes = [tot_pos, tot_neg, tot_neu]
explode = (0, 0.1, 0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

