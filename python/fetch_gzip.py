from lxml import html
import requests
import os

url = "https://dumps.wikimedia.org/other/pagecounts-raw/2015/2015-12/"
page = requests.get(url)
tree = html.fromstring(page.content)

pcounts = tree.xpath('//ul/li/a/@href')

def fetch_gz(fname):
    # partial page count url
    ppc_url = 'https://dumps.wikimedia.org/other/pagecounts-raw/2015/2015-12/'
    print("fetching: "+ fname)
    pc_url = ppc_url+fname
    r = requests.get(pc_url, stream=True)
    with open(fname, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
            f.flush()

from hdfs3 import HDFileSystem
hdfs = HDFileSystem()

def fetch_gz(fname):
    # partial page count url
    ppc_url = 'https://dumps.wikimedia.org/other/pagecounts-raw/2015/2015-12/'
    print("fetching: "+ fname)
    pc_url = ppc_url+fname
    r = requests.get(pc_url, stream=True)
    fname_path = os.path.join('/tmp','wiki',fname)
    with hdfs.open(fname_path, 'w') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

from distributed import Executor
executor = Executor('127.0.0.1:8786')

executor.restart()

x = executor.map(fetch_gz, pcounts)

x

from distributed import progress

progress(x)

hdfs.ls('/tmp/wiki/')

from gzip import GzipFile
with hdfs.open('/tmp/wiki/pagecounts-20151229-070000.gz') as f:
    g = GzipFile(fileobj=f)
    print(g.read(1000))

with hdfs.open('/tmp/wiki/projectcounts-20151218-000000') as f:
    f.seek(10000)
    print(f.read(1000))

hdfs.df()

hdfs.exists('/tmp/wiki/pagecounts-20151229-070000')

