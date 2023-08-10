import mechanize
from time import sleep
import os

data_dir = '/root/data/hackathon/building_massa_dataset/'

train = 'https://www.cs.toronto.edu/~vmnih/data/mass_buildings/train/sat/index.html'
val = 'https://www.cs.toronto.edu/~vmnih/data/mass_buildings/valid/sat/index.html'
test = 'https://www.cs.toronto.edu/~vmnih/data/mass_buildings/test/sat/index.html'

urls = [test]

for url in urls:
    #Make a Browser (think of this as chrome or firefox etc)
    br = mechanize.Browser()

    # Open your site
    br.open(url)

    f=open("source.html","w")
    f.write(br.response().read()) #can be helpful for debugging maybe

    filetypes=[".tiff"] #you will need to do some kind of pattern matching on your files
    myfiles=[]
    for l in br.links(): #you can also iterate through br.forms() to print forms on the page!
        for t in filetypes:
            if t in str(l): #check if this link has the file extension we want (you may choose to use reg expressions or something)
                myfiles.append(l.url)
    # print(myfiles)

outF = open(os.path.join('/root/data/hackathon/building_massa_dataset/images/test/', "download_images.txt"), "w")
for line in myfiles:
    # write line to output file
    outF.write(line)
    outF.write("\n")
outF.close()

train = 'https://www.cs.toronto.edu/~vmnih/data/mass_buildings/train/map/index.html'
val = 'https://www.cs.toronto.edu/~vmnih/data/mass_buildings/valid/map/index.html'
test = 'https://www.cs.toronto.edu/~vmnih/data/mass_buildings/test/map/index.html'

urls = [test]

for url in urls:
    #Make a Browser (think of this as chrome or firefox etc)
    br = mechanize.Browser()

    # Open your site
    br.open(url)

    f=open("source.html","w")
    f.write(br.response().read()) #can be helpful for debugging maybe

    filetypes=[".tif"] #you will need to do some kind of pattern matching on your files
    myfiles=[]
    for l in br.links(): #you can also iterate through br.forms() to print forms on the page!
        for t in filetypes:
            if t in str(l): #check if this link has the file extension we want (you may choose to use reg expressions or something)
                myfiles.append(l.url)
    # print(myfiles)

outF = open(os.path.join('/root/data/hackathon/building_massa_dataset/labels/test/', "download_images.txt"), "w")
for line in myfiles:
    # write line to output file
    outF.write(line)
    outF.write("\n")
outF.close()



