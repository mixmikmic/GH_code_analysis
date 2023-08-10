from bs4 import BeautifulSoup
import requests
from IPython.display import HTML
import re
import pickle

from IPython.display import clear_output

page = requests.get('https://www.engadget.com/2017/03/24/razers-paid-to-play-program-bribes-gamers-to-use-its-cortex-s/')
soup = BeautifulSoup(page.content, 'html.parser')

soup.prettify()

html = ""
for n in soup.findAll('div', attrs={"class":"article-text"}):
    html += str(n)
    
HTML(html)

for a in soup.findAll('a'):
    mask = re.compile('https:\/\/www.engadget.com\/\d\d\d\d\/\d\d\/\d\d\/(.*)');
    # print(mask.match(a['href'] ) )
    if mask.match(a['href']) is not None:
        print(a)

class Scrape():
    def __init__(self, folder):
        self.cache_file_name = folder + ".scrape.progress"
        self.post_url_mask = re.compile('https:\/\/www.engadget.com\/(\d\d\d\d\/\d\d\/\d\d)\/([^?\/:]*)(.*)');
        self.folder = folder
        self.active = []
        self.done = []
        self.i = 0
        
        
    def scrape(self, url):
        if url in self.done: return
        self.done.append(url)
        
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        
        if self.post_url_mask.match(url):
            html = ""
            for n in soup.findAll('div', attrs={"class":"article-text"}):
                html += str(n)

            match = self.post_url_mask.match(url)
            title = match.group(2)
            date = match.group(1)

            print(date + ":  " + title)

            with open(self.folder + title + '.txt', 'w+') as f:
                soup = BeautifulSoup(html.replace('</p>', '\n\n</p>').replace('  ', ' '), 'html.parser')
                f.write(soup.getText())
        
        for a in soup.findAll('a'):
            try: 
                if mask.match(a['href']) is not None:
                    self.add_link(a['href'])
            except KeyError:
                print('Warning: anchor has empty href.')
            
    def craw(self, url, once=False):
        if url is not None:
            self.active.append(url)

        while len(self.active) > 0:
            self.i += 1
            url = self.active.pop(0)
            self.scrape(url)
            self.done.append(url)
            
            if once:
                break
                
            if self.i%50 == 49:
                clear_output(wait=True)
                print("... #{:d}".format(self.i))
        
    def add_link(self, link):
        if link in self.done or link in self.active:
            return
        self.active.append(link)
            
    def __enter__(self, *args):
        try:
            with open(self.cache_file_name, 'rb') as f:
                
                print(self.cache_file_name)
                cache = pickle.load(f)
                self.active = cache['active'] or []
                self.done = cache['done'] or []
                self.i = cache['i'] or 0
                
        except FileNotFoundError:
            print('no progress file found')
        except EOFError:
            print('Warning: file is empty')
        except AttributeError as err:
            print('cache does not have key ' + str(err))
        
        return self
            
    def __exit__(self, *args):
        print('scrape has exited')
        with open(self.cache_file_name, 'wb') as f:
            print(self.active, self.done)
            pickle.dump({
                "active": self.active,
                "done": self.done,
                "i": self.i
            }, f)
        print('scrape state has been saved')
        
with Scrape('./engadget_data/') as s:
    
    s.craw('https://www.engadget.com')
    #s.craw('https://www.engadget.com/2010/06/23/google-wins-youtube-copyright-case-against-viacom/')

# with open("./.scrape.progress", 'wb') as f:
#     cache = pickle.dump({"active":[], 'done':[]}, f)
#     print(cache)

with open("./engadget_data/.scrape.progress", 'rb') as f:
    cache = pickle.load(f)
    print(cache)



