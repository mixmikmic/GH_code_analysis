#import urllib2 #uncomment for Python 2
import urllib.request #uncomment for Python 3
import json
from IPython.core.display import Image, display

#paste productIds as string
productIds = '7544181 7464260 7620565 7462148 7594693'

productIds = productIds.split()

for productId in productIds:

    #uncomment for Python2
    #data = json.load(urllib2.urlopen('http://es-sor-recs-cs.cloudapp.net:29200/products2/v1/'+str(productId)))

    #uncomment for Python3
    data = json.load(urllib.request.urlopen('http://es-sor-recs-cs.cloudapp.net:29200/products2/v1/'+str(productId)))

    image_url = data['_source']['imageloc']

    display(Image(image_url, width=500, unconfined=True))



