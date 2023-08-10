import bs4 as bs
import urllib.request

sauce = urllib.request.urlopen('https://elacsoft.github.io/').read()

soup = bs.BeautifulSoup(sauce, 'lxml')
print(soup)

print(soup.title.string)

for urls in soup.find_all('a'):
    print(urls)
    print("-----")
    print(urls.get('href'))
    print('\n\n')

for projects in soup.find_all('form'):
    print(projects.a.input.get('value'))

sauce = urllib.request.urlopen("https://pixabay.com/en/photos/cat/").read()
soup = bs.BeautifulSoup(sauce, 'lxml')
count = 0
for imgs in soup.find_all('img'):
    print(imgs.get('src'))
    count += 1
    if count == 5:
        break
    else:
        pass

sauce = urllib.request.urlopen("https://pixabay.com/en/photos/cat/").read()
soup = bs.BeautifulSoup(sauce, 'lxml')
count = 0
i = 1
for imgs in soup.find_all('img'):
    #print(imgs.get('src'))
    temp = imgs.get('src')
    count += 1
    if temp[:1] == "/":
        image = "https://pixabay.com/en/photos/cat" + temp
    else:
        image = temp
    print(image)
    nametemp = imgs.get('alt')
    if len(nametemp) == 0:
        filename = str(i)
        i+=1
    else:
        filename = nametemp
    imagefile = open(filename + '.jpeg','wb')
    imagefile.write(urllib.request.urlopen(image).read())
    imagefile.close()
    if count == 5:
        break
    else:
        pass



