import matplotlib.pyplot as plt
import numpy as np
from testing_utilities import *
import requests

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

image_name = "fboylu/kerastf-gpu"

get_ipython().run_cell_magic('bash', '--bg -s "$image_name"', 'nvidia-docker run -p 80:80 $1')

get_ipython().system("curl 'http://0.0.0.0:80/'")

get_ipython().system("curl 'http://0.0.0.0:80/version'")

IMAGEURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Lynx_lynx_poing.jpg/220px-Lynx_lynx_poing.jpg"

plt.imshow(to_img(IMAGEURL))

jsonimg = img_url_to_json(IMAGEURL)
jsonimg[:100] 

headers = {'content-type': 'application/json'}
get_ipython().run_line_magic('time', "r = requests.post('http://0.0.0.0:80/score', data=jsonimg, headers=headers)")
print(r)
r.json()

images = ('https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Lynx_lynx_poing.jpg/220px-Lynx_lynx_poing.jpg',
          'https://upload.wikimedia.org/wikipedia/commons/3/3a/Roadster_2.5_windmills_trimmed.jpg',
          'http://www.worldshipsociety.org/wp-content/themes/construct/lib/scripts/timthumb/thumb.php?src=http://www.worldshipsociety.org/wp-content/uploads/2013/04/stock-photo-5495905-cruise-ship.jpg&w=570&h=370&zc=1&q=100',
          'http://yourshot.nationalgeographic.com/u/ss/fQYSUbVfts-T7pS2VP2wnKyN8wxywmXtY0-FwsgxpiZv_E9ZfPsNV5B0ER8-bOdruvNfMD5EbP4SznWz4PYn/',
          'https://cdn.arstechnica.net/wp-content/uploads/2012/04/bohol_tarsier_wiki-4f88309-intro.jpg',
          'http://i.telegraph.co.uk/multimedia/archive/03233/BIRDS-ROBIN_3233998b.jpg')

url = 'http://0.0.0.0:80/score'
results = [requests.post(url, data=img_url_to_json(img), headers=headers) for img in images]

plot_predictions_dict(images, results)

image_data = list(map(img_url_to_json, images)) # Retrieve the images and data

timer_results = list()
for img in image_data:
    res = get_ipython().run_line_magic('timeit', '-r 1 -o -q requests.post(url, data=img, headers=headers)')
    timer_results.append(res.best)

timer_results

print('Average time taken: {0:4.2f} ms'.format(10**3 * np.mean(timer_results)))

get_ipython().run_cell_magic('bash', '', 'docker stop $(docker ps -q)')

