import asyncio
import json
import random
import urllib.request
from timeit import default_timer

import aiohttp
import matplotlib.pyplot as plt
import testing_utilities
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')

NUMBER_OF_REQUESTS = 100  # Total number of requests
CONCURRENT_REQUESTS = 4   # Number of requests at a time

service_json = get_ipython().getoutput('kubectl get service azure-dl -o json')
service_dict = json.loads(''.join(service_json))
app_url = service_dict['status']['loadBalancer']['ingress'][0]['ip']

scoring_url = 'http://{}/score'.format(app_url)
version_url = 'http://{}/version'.format(app_url)

get_ipython().system('curl $version_url # Reports the Tensorflow Version')

IMAGEURL = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Lynx_lynx_poing.jpg/220px-Lynx_lynx_poing.jpg"
plt.imshow(testing_utilities.to_img(IMAGEURL))

def gen_variations_of_one_image(num, label='image'):
    out_images = []
    img = testing_utilities.to_img(IMAGEURL).convert('RGB')
    # Flip the colours for one-pixel
    # "Different Image"
    for i in range(num):
        diff_img = img.copy()
        rndm_pixel_x_y = (random.randint(0, diff_img.size[0]-1), 
                          random.randint(0, diff_img.size[1]-1))
        current_color = diff_img.getpixel(rndm_pixel_x_y)
        diff_img.putpixel(rndm_pixel_x_y, current_color[::-1])
        b64img = testing_utilities.to_base64(diff_img)
        out_images.append(json.dumps({'input':{label:'\"{0}\"'.format(b64img)}}))
    return out_images

url_list = [[scoring_url, jsonimg] for jsonimg in gen_variations_of_one_image(NUMBER_OF_REQUESTS)]

def decode(result):
    return json.loads(result.decode("utf-8"))

async def fetch(url, session, data, headers):
    start_time = default_timer()
    async with session.request('post', url, data=data, headers=headers) as response:
        resp = await response.read()
        elapsed = default_timer() - start_time
        return resp, elapsed

async def bound_fetch(sem, url, session, data, headers):
    # Getter function with semaphore.
    async with sem:
        return await fetch(url, session, data, headers)

async def await_with_progress(coros):
    results=[]
    for f in tqdm(asyncio.as_completed(coros), total=len(coros)):
        result = await f
        results.append((decode(result[0]),result[1]))
    return results

async def run(url_list, num_concurrent=CONCURRENT_REQUESTS):
    headers = {'content-type': 'application/json'}
    tasks = []
    # create instance of Semaphore
    sem = asyncio.Semaphore(num_concurrent)

    # Create client session that will ensure we dont open new connection
    # per each request.
    async with aiohttp.ClientSession() as session:
        for url, data in url_list:
            # pass Semaphore and session to every POST request
            task = asyncio.ensure_future(bound_fetch(sem, url, session, data, headers))
            tasks.append(task)
        return await await_with_progress(tasks)

loop = asyncio.get_event_loop()
start_time = default_timer()
complete_responses = loop.run_until_complete(asyncio.ensure_future(run(url_list, num_concurrent=CONCURRENT_REQUESTS)))
elapsed = default_timer() - start_time
print('Total Elapsed {}'.format(elapsed))
print('Avg time taken {0:4.2f} ms'.format(1000*elapsed/len(url_list)))

complete_responses[:3]

num_succesful=[i[0]['result'][0]['image'][0][0] for i in complete_responses].count('n02127052 lynx, catamount')
print('Succesful {} out of {}'.format(num_succesful, len(url_list)))

# Example response
plt.imshow(testing_utilities.to_img(IMAGEURL))
complete_responses[0]

