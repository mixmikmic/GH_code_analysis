import requests
from requests_throttler import BaseThrottler

bt = BaseThrottler(name='base-throttler', delay=1.5)
request = requests.Request(method='GET', url='http://www.google.com')
reqs = [request for i in range(0, 5)]

bt.start()
throttled_requests = bt.multi_submit(reqs)
bt.shutdown()

responses = [tr.response for tr in throttled_requests]



