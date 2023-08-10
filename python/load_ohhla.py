with open('../data/ohhla/train/www.ohhla.com/anonymous/j_live/allabove/satisfy.jlv.txt.html', 'r') as f:
    # we use read().splitlines() instead of readlines() to skip newline characters
    lines = f.read().splitlines()
    
lines

def find_lyrics(lines):
    filtered = []
    in_pre = False
    for line in lines:
        if '<pre>' in line:
            in_pre = True
            filtered.append(line.replace("<pre>",""))
        elif '</pre>' in line:
            in_pre = False
            filtered.append(line.replace("</pre>",""))
        elif in_pre:
            filtered.append(line)
    return filtered[6:]
    
lyrics = find_lyrics(lines)
lyrics[:10]

string = '[BAR]' + '[/BAR][BAR]'.join(lyrics) + '[/BAR]'
string[:500]

def load_song(file_name):
    def load_raw(encoding):
        with open(file_name, 'r',encoding=encoding) as f:
            # we use read().splitlines() instead of readlines() to skip newline characters
            lines = f.read().splitlines()   
            # some files are pure txt files for which we don't need to extract the lyrics 
            lyrics = find_lyrics(lines) if file_name.endswith('html') else lines[5:]
            string = '[BAR]' + '[/BAR][BAR]'.join(lyrics) + '[/BAR]'
            return string
    try:
        return load_raw('utf-8')
    except UnicodeDecodeError:
        try:
            return load_raw('cp1252')
        except UnicodeDecodeError:
            print("Could not load " + file_name)
            return ""

        
    
song = load_song('../data/ohhla/train/www.ohhla.com/anonymous/j_live/allabove/satisfy.jlv.txt.html')
song[:500]

from os import listdir
from os.path import isfile, join

def load_album(path):
    # we filter out directories, and files that don't look like song files in OHHLA.
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and 'txt' in f]
    lyrics = [load_song(f) for f in onlyfiles]
    return lyrics

songs = load_album('../data/ohhla/train/www.ohhla.com/anonymous/j_live/SPTA/')
[len(s) for s in songs]

def load_albums(album_paths):
    return [song 
            for path in album_paths 
            for song in load_album(path)]

top_dir = '../data/ohhla/train/www.ohhla.com/anonymous/'
j_live = [
    top_dir + '/j_live/allabove/',
    top_dir + '/j_live/bestpart/'
]
len(load_albums(j_live))

import re
token = re.compile("\[BAR\]|\[/BAR\]|[\w-]+|'m|'t|'ll|'ve|'d|'s|\'")
def words(docs):
    return [word 
            for doc in docs 
            for word in token.findall(doc)]
song_words = words(songs)
song_words[:20]


def load_all_songs(path):
    only_files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and 'txt' in f]
    only_paths = [join(path, f) for f in listdir(path) if not isfile(join(path, f))]
    lyrics = [load_song(f) for f in only_files]
    sub_songs = [song for sub_path in only_paths for song in load_all_songs(sub_path)]
    return lyrics + sub_songs

len(load_all_songs("../data/ohhla/train/www.ohhla.com/anonymous/j_live/"))

