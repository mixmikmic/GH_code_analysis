class A(object):
    def __enter__(self):
        self.f = "Init"
        return self
    def __exit__(self,type,value,traceback):
        if self.f == "Init":
            print('Destroying f')
            self.f = 'Destroyed'
            print(self.f)
            #print(self,type,value,tb)
        #exit should not return anything
    
with A() as a:
    print(a.f)

import time

class Timer(object):
    def __enter__(self):
        self.start = time.clock()
        return self
    def __exit__(self,*args):
        self.end = time.clock()
        self.interval = self.end - self.start
        
import http.client

with Timer() as t:
    conn = http.client.HTTPConnection('google.com')
    conn.request('GET', '/')
    
print('Request took like {0}'.format(t.interval))

from contextlib import contextmanager

@contextmanager
def tag(name):
    print("<%s>" % name)
    yield
    print("</%s>" % name)
    
with tag("h2"):
    print("Sparta")

from contextlib import ContextDecorator
import logging

logging.basicConfig(level=logging.INFO)

class track_entry_and_exit(ContextDecorator):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        logging.info('Entering: {}'.format(self.name))

    def __exit__(self, exc_type, exc, exc_tb):
        logging.info('Exiting: {}'.format(self.name))
 



