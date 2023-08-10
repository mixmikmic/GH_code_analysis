from testing_utilities import *
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

app_url = 'mscntkacsagents.southcentralus.cloudapp.azure.com'
app_id = '/cntkresnet'

scoring_url = 'http://{}/score'.format(app_url)
version_url = 'http://{}/version'.format(app_url)

get_ipython().system('curl $version_url # Reports the CNTK Version')

IMAGEURL = "https://www.britishairways.com/assets/images/information/about-ba/fleet-facts/airbus-380-800/photo-gallery/240x295-BA-A380-exterior-2-high-res.jpg"

plt.imshow(to_img(IMAGEURL))

headers = {'content-type': 'application/json','X-Marathon-App-Id': app_id}
jsonimg = img_url_to_json(IMAGEURL)
r = requests.post(scoring_url, data=jsonimg, headers=headers) # Run the request twice since the first time takes a 
                                                              # little longer due to the loading of the model
get_ipython().magic('time r = requests.post(scoring_url, data=jsonimg, headers=headers)')
r.json()

images = ('https://www.britishairways.com/assets/images/information/about-ba/fleet-facts/airbus-380-800/photo-gallery/240x295-BA-A380-exterior-2-high-res.jpg',
          'https://upload.wikimedia.org/wikipedia/commons/3/3a/Roadster_2.5_windmills_trimmed.jpg',
          'http://www.worldshipsociety.org/wp-content/themes/construct/lib/scripts/timthumb/thumb.php?src=http://www.worldshipsociety.org/wp-content/uploads/2013/04/stock-photo-5495905-cruise-ship.jpg&w=570&h=370&zc=1&q=100',
          'http://yourshot.nationalgeographic.com/u/ss/fQYSUbVfts-T7pS2VP2wnKyN8wxywmXtY0-FwsgxpiZv_E9ZfPsNV5B0ER8-bOdruvNfMD5EbP4SznWz4PYn/',
          'https://cdn.arstechnica.net/wp-content/uploads/2012/04/bohol_tarsier_wiki-4f88309-intro.jpg',
          'http://i.telegraph.co.uk/multimedia/archive/03233/BIRDS-ROBIN_3233998b.jpg')

results = [requests.post(scoring_url, data=img_url_to_json(img), headers=headers) for img in images]

plot_predictions(images, results)

timer_results = list()
for img in images:
    res = get_ipython().magic('timeit -r 1 -o -q requests.post(scoring_url, data=img_url_to_json(img), headers=headers)')
    timer_results.append(res.best)

timer_results

print('Average time taken: {0:4.2f} ms'.format(10**3 * np.mean(timer_results)))

