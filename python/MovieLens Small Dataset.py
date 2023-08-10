import os
import requests
import zipfile

DATA_DIR = 'movielens'
DATASET_URL = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
DATASET_FILENAME = DATASET_URL.split('/')[-1]
DATASET_PACKAGE = os.path.join(DATA_DIR, DATASET_FILENAME)
DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME[:-4])

if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)
    
if not os.path.isfile(DATASET_PACKAGE):
    print('Downloading {}...'.format(DATASET_FILENAME))
    r = requests.get(DATASET_URL, stream=True)
    with open(DATASET_PACKAGE, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print('Done!')

if not os.path.isdir(DATASET_PATH):
    print('Unpacking {}...'.format(DATASET_PACKAGE))
    with zipfile.ZipFile(DATASET_PACKAGE, 'r') as f:
        f.extractall(DATA_DIR)
    print('Done!')

LINKS_CSV = os.path.join(DATASET_PATH, 'links.csv')

with open(LINKS_CSV, 'r') as f:
    for _ in range(10):
        print(f.readline().strip())

import requests

movie_id, imdb_id = '1', '0114709'
r = requests.get('http://www.omdbapi.com/?i=tt{}&plot=full'.format(imdb_id))
movie_data = r.json()
movie_plot = movie_data["Plot"]

print('{}\n\n{}'.format(movie_id, movie_plot))

get_ipython().run_cell_magic('time', '', '\nimport asyncio\nimport collections\nimport csv\nimport functools\nimport requests\n\nLINKS_CSV = \'movielens/ml-latest-small/links.csv\'\nPLOTS_CSV = \'movielens/ml-latest-small/plots.csv\'\n\nMOVIE_DATA_URL = \'http://www.omdbapi.com\'\n\nMovieRequest = collections.namedtuple(\'MovieRequest\', [\'movie_id\', \'imdb_id\', \'tries\'])\nMovieData = collections.namedtuple(\'MovieData\', [\'movie_id\', \'plot\'])\n\nrequest_queue = asyncio.Queue(100)\nfail_queue = asyncio.Queue(100)\ndata_queue = asyncio.Queue(100)\n\npipeline_active = asyncio.Event()\nproducer_done = asyncio.Event()\n\nasync def pipeline():\n    print(\'Pipeline started...\')\n    pipeline_active.set()\n    await asyncio.sleep(2)\n    await producer_done.wait()\n    await request_queue.join()\n    await fail_queue.join()\n    await data_queue.join()\n    pipeline_active.clear()\n    await asyncio.sleep(2)\n    print(\'Pipeline done!\')\n\n\nasync def request_producer(filename, request_queue):\n    await pipeline_active.wait()\n    print(\'Request producer started...\')\n    N = 3 * request_queue.maxsize // 4\n    with open(filename, newline=\'\') as f:\n        reader = csv.reader(f)\n        next(reader) # skip header\n        for n, (movie_id, imdb_id, tmdb_id) in enumerate(reader):\n            while request_queue.qsize() > N:\n                await asyncio.sleep(2)\n            await request_queue.put(MovieRequest(movie_id, imdb_id, 0))\n            if (n+1) % 1000 == 0:\n                print(\'Requests... {:,d}\'.format(n+1))\n    producer_done.set()\n    print(\'Request producer done!\')\n\n\ns = requests.Session()\nadapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=3)\ns.mount(MOVIE_DATA_URL, adapter)\ns.params[\'plot\'] = \'full\'\n\ndef load_movie_data(imdb_id, timeout=5):\n    r = s.get(MOVIE_DATA_URL, params={\'i\': \'tt\' + imdb_id}, timeout=timeout)\n    return r.json()\n\n\nasync def request_worker(name, loop, request_queue, data_queue, fail_queue, retries=3):\n    await pipeline_active.wait()\n    print(\'Request worker {} started...\'.format(name))\n    n, t, f = 0, 0, 0\n    while pipeline_active.is_set():\n        try:\n            req = await asyncio.wait_for(request_queue.get(), 1)\n        except asyncio.TimeoutError:\n            #print(\'Request worker {} timeout...\'.format(name))\n            continue\n        try:\n            data = await loop.run_in_executor(None, functools.partial(load_movie_data, req.imdb_id))\n            n += 1\n            await data_queue.put(MovieData(req.movie_id, data["Plot"]))\n        except Exception as e:\n            #print(\'Request worker {} fail...\'.format(name))\n            #print(e)\n            if req.tries < retries:\n                t += 1\n                await request_queue.put(req._replace(tries=req.tries+1))\n            else:\n                f += 1\n                await fail_queue.put(req)\n        request_queue.task_done()\n    print(\'Request worker {} done! {:,d} requests, {:,d} retries, {:,d} fails\'.format(name, n, t, f))\n\n\nasync def csv_writer(filename, data_queue):\n    await pipeline_active.wait()\n    print(\'CSV Writer (filename={}) started...\'.format(filename))\n    n = 0\n    with open(filename, \'w\', newline=\'\') as f:\n        writer = csv.writer(f)\n        writer.writerow([\'movieId\', \'plot\'])\n\n        while pipeline_active.is_set():\n            try:\n                data = await asyncio.wait_for(data_queue.get(), 1)\n            except asyncio.TimeoutError:\n                #print(\'CSV Plot timeout...\')\n                continue\n            movie_id, movie_plot = data\n            writer.writerow([movie_id, movie_plot])\n            n += 1\n            data_queue.task_done()\n            \n            if n % 1000 == 0:\n                print(\'CSV rows... {:,d}\'.format(n))\n\n    print(\'CSV Writer done! {:,d} rows\'.format(n))\n\n\nasync def fail_sink(fail_queue):\n    await pipeline_active.wait()\n    print(\'Fail Sink started...\')\n    n = 0\n    while pipeline_active.is_set():\n        try:\n            fail = await asyncio.wait_for(fail_queue.get(), 1)\n        except asyncio.TimeoutError:\n            #print(\'Fail Sink timeout...\')\n            continue\n        n += 1\n        if n <= 100:\n            print(\'Lost: {}\'.format(fail))\n        if n % 1000 == 0:\n            print(\'Lost: {:,d}\'.format(n))\n        fail_queue.task_done()\n    print(\'Fail sink done! {:,d} fails\'.format(n))\n\n    \nloop = asyncio.get_event_loop()\n\nloop.create_task(fail_sink(fail_queue))\n\nloop.create_task(csv_writer(PLOTS_CSV, data_queue))\n\nfor i in range(5):\n    loop.create_task(request_worker(str(i+1), loop, request_queue, data_queue, fail_queue))\n\nloop.create_task(request_producer(LINKS_CSV, request_queue))\n\nt = loop.create_task(pipeline())\n\nloop.run_until_complete(t)\n\nprint(request_queue.qsize())\nprint(fail_queue.qsize())\nprint(data_queue.qsize())')

