import requests
import concurrent.futures

urlList = []
# Open a url text file: './man_sport.txt'
filename = 'man_sport.txt'
# Read file
with open(filename) as f:
    for line in f:
        urlList += [line]    

# Crawl images and save
# Path is the target directory to save images
def loadImage(url, path):
    res = requests.get(url)
    if res.status_code == 200:
        path = path + url.split('/')[-1].split('\n')[0]
        with open(path, 'wb') as f:
            f.write(res.content)

# Multi-thread to speed up 
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    for url in urlList:
        try:
            executor.submit(loadImage, url, 'male/')
        except Exception as exc:
            print(exc)

