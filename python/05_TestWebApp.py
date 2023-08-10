import matplotlib.pyplot as plt
import numpy as np
from testing_utilities import *
import requests

get_ipython().run_line_magic('matplotlib', 'inline')

service_json = get_ipython().getoutput('kubectl get service azure-dl -o json')
service_dict = json.loads(''.join(service_json))
app_url = service_dict['status']['loadBalancer']['ingress'][0]['ip']

scoring_url = 'http://{}/score'.format(app_url)
version_url = 'http://{}/version'.format(app_url)
health_url = 'http://{}/'.format(app_url)

get_ipython().system('curl $health_url')

get_ipython().system('curl $version_url # Reports the tensorflow version')

IMAGEURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Lynx_lynx_poing.jpg/220px-Lynx_lynx_poing.jpg"

plt.imshow(to_img(IMAGEURL))

jsonimg = img_url_to_json(IMAGEURL)
headers = {'content-type': 'application/json'}
r = requests.post(scoring_url, data=jsonimg, headers=headers) # Run the request twice since the first time takes a 
get_ipython().run_line_magic('time', 'r = requests.post(scoring_url, data=jsonimg, headers=headers) # little longer due to the loading of the model')
print(r)
r.json()

images = ('https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Lynx_lynx_poing.jpg/220px-Lynx_lynx_poing.jpg',
          'https://upload.wikimedia.org/wikipedia/commons/3/3a/Roadster_2.5_windmills_trimmed.jpg',
          'http://www.worldshipsociety.org/wp-content/themes/construct/lib/scripts/timthumb/thumb.php?src=http://www.worldshipsociety.org/wp-content/uploads/2013/04/stock-photo-5495905-cruise-ship.jpg&w=570&h=370&zc=1&q=100',
          'http://yourshot.nationalgeographic.com/u/ss/fQYSUbVfts-T7pS2VP2wnKyN8wxywmXtY0-FwsgxpiZv_E9ZfPsNV5B0ER8-bOdruvNfMD5EbP4SznWz4PYn/',
          'https://cdn.arstechnica.net/wp-content/uploads/2012/04/bohol_tarsier_wiki-4f88309-intro.jpg',
          'http://i.telegraph.co.uk/multimedia/archive/03233/BIRDS-ROBIN_3233998b.jpg')

results = [requests.post(scoring_url, data=img_url_to_json(img), headers=headers) for img in images]

plot_predictions_dict(images, results)

image_data = list(map(img_url_to_json, images)) # Retrieve the images and data

timer_results = list()
for img in image_data:
    res = get_ipython().run_line_magic('timeit', '-r 1 -o -q requests.post(scoring_url, data=img, headers=headers)')
    timer_results.append(res.best)

timer_results

print('Average time taken: {0:4.2f} ms'.format(10**3 * np.mean(timer_results)))

