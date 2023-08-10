import requests
import bs4
import pandas

r = requests.get('http://www.imdb.com/list/ls058982125/')

mypage = bs4.BeautifulSoup(r.text, 'lxml')

mymovies = mypage.find_all('div', attrs={'class': 'lister-item-content'})

#mymovies[0]

movies = []
for mymovie in mymovies:
    mytitle = mymovie.find('h3').find('a').text
    myrating = mymovie.find(
        'div', attrs={'class': 'ipl-rating-star small'}
    ).find(
        'span', attrs={'class': 'ipl-rating-star__rating'}
    ).text
    myrating = float(myrating)
    myruntime = mymovie.find(
        'span', attrs={'class': 'runtime'}
    ).text
    movies.append([mytitle, myrating, myruntime])

movies

df = pandas.DataFrame(movies, columns=['title', 'rating', 'runtime'])

df

df.to_csv('imdb.com/imdb.com.csv')

get_ipython().system('ls imdb.com/imdb.com.csv')

get_ipython().system('head -n 5 imdb.com/imdb.com.csv')

import datetime
print('Last updated:', datetime.datetime.now())



