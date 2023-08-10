import urllib.request
import matplotlib.pyplot as plt
import aiohttp
import asyncio
import json
import random
from io import BytesIO
from PIL import Image, ImageOps
get_ipython().magic('matplotlib inline')

print(aiohttp.__version__)  # 2.0.3

# SET
NUM = 100
CONC = 50
SERVER_URL = 'http://wincntkdemo.azurewebsites.net/api/uploader'

def get_one_image(url):
    urllib.request.urlretrieve(url, "test.jpg")
    plt.axis('off')
    plt.imshow(Image.open('test.jpg'))   

def gen_variations_of_one_image(num):
    out_images = []
    imagefile = open('test.jpg', 'rb')
    img = Image.open(BytesIO(imagefile.read())).convert('RGB')
    img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
    # Flip the colours for one-pixel
    # "Different Image"
    for i in range(num):
        diff_img = img.copy()
        rndm_pixel_x_y = (random.randint(0, diff_img.size[0]-1), 
                          random.randint(0, diff_img.size[1]-1))
        current_color = diff_img.getpixel(rndm_pixel_x_y)
        diff_img.putpixel(rndm_pixel_x_y, current_color[::-1])
        # Turn image into IO
        ret_imgio = BytesIO()
        diff_img.save(ret_imgio, 'PNG')
        out_images.append(ret_imgio.getvalue())
    return out_images

get_one_image("https://i.ytimg.com/vi/96xC5JIkIpQ/maxresdefault.jpg")

images = gen_variations_of_one_image(NUM)

url_list = [[SERVER_URL, {'imagefile':pic}] for pic in images]

def handle_req(data):
    return json.loads(data.decode('utf-8'))
 
def chunked_http_client(num_chunks, s):
    # Use semaphore to limit number of requests
    semaphore = asyncio.Semaphore(num_chunks)
    @asyncio.coroutine
    def http_get(dta):
        nonlocal semaphore
        with (yield from semaphore):
            url, img = dta
            response = yield from s.request('post', url, data=img)
            body = yield from response.content.read()
            yield from response.wait_for_close()
        return body
    return http_get
 
def run_experiment(urls, _session):
    http_client = chunked_http_client(num_chunks=CONC, s=_session)
    # http_client returns futures, save all the futures to a list
    tasks = [http_client(url) for url in urls]
    rsponses = []
    # wait for futures to be ready then iterate over them
    for future in asyncio.as_completed(tasks):
        data = yield from future
        try:
            out = handle_req(data)
            rsponses.append(out)
        except Exception as err:
            print("Error {0}".format(err))
    return rsponses

get_ipython().run_cell_magic('time', '', "# Expect to see some 'errors' meaning requests are expiring on 'queue'\n# i.e. we can't increase concurrency any more\nwith aiohttp.ClientSession() as session:  # We create a persistent connection\n    loop = asyncio.get_event_loop()\n    complete_responses = loop.run_until_complete(run_experiment(url_list, session)) ")

len(complete_responses)
print(complete_responses[:5])

# Total responses
len(complete_responses)

# Successful responses
complete_responses.count(complete_responses[1])

# In this example
print(86/NUM, " seconds per image")

