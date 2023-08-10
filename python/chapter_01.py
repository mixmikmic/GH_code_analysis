stack = [3, 4, 5]
stack.append(6)
stack.append(7)
stack

stack.pop()

stack

stack.pop()

stack

stack.pop()

stack

import pandas as pd
import matplotlib
import matplotlib.dates as mdates

get_ipython().magic('matplotlib inline')
matplotlib.style.use('ggplot')

df = pd.read_csv("djia.csv", comment="#", parse_dates=[0], index_col=0, names=["date", "djia"])
plt = df.plot(figsize=(16, 10))
plt.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
min_x = df.index.min()
max_x = df.index.max()
plt.set_xlim([min_x, max_x])
ticks = pd.date_range(start=min_x, end=max_x, freq='10A')
_ = plt.set_xticks(ticks)

def simple_stock_span(quotes):
    spans = []
    for i in range(len(quotes)):
        k = 1
        span_end = False
        while i - k >= 0 and not span_end:
            if quotes[i - k] <= quotes[i]:
                k += 1
            else:
                span_end = True
        spans.append(k)
    return spans

def read_quotes(filename):
    quotes = []
    with open(filename) as quotes_file:
        for line in quotes_file:
            if line.startswith('#'):
                continue
            parts = line.split(',')
            quotes.append(float(parts[-1]))
    return quotes

quotes = read_quotes("djia.csv")

len(quotes)

spans = simple_stock_span(quotes)
print(quotes[-1630:-1620])

import operator

max_index, max_value = max(enumerate(spans), key=operator.itemgetter(1))

with open('djia.csv') as quotes_file:
  for i, line in enumerate(quotes_file):
    if i == max_index:
      print(max_value, line)
      break

import time

min_time_taken = 10**10

for i in range(10):
    time_start = time.time()
    spans = simple_stock_span(quotes)
    time_end = time.time()
    time_diff = time_end - time_start
    if time_diff < min_time_taken:
        min_time_taken = time_diff
        
print("Best time was",  1000 * min_time_taken, "milliseconds.")
    

def stack_stock_span(quotes):
    spans = [1]
    s = []
    s.append(0)
    for i in range(1, len(quotes)):
        while len(s) != 0 and quotes[s[-1]] <= quotes[i]:
            s.pop()
        if len(s) == 0:
            spans.append(i+1)
        else:
            spans.append(i - s[-1])
        s.append(i)
    return spans

spans = stack_stock_span(quotes)
print(spans[-10:])

min_time_taken = 10**10

for i in range(10):
    time_start = time.time()
    spans = stack_stock_span(quotes)
    time_end = time.time()
    time_diff = time_end - time_start
    if time_diff < min_time_taken:
        min_time_taken = time_diff
        
print("Best time was",  1000 * min_time_taken, "milliseconds.")

import time

def simple_stock_span(quotes):
    spans = []
    for i in range(len(quotes)):
        k = 1
        span_end = False
        while i - k >= 0 and not span_end:
            if quotes[i - k] <= quotes[i]:
                k += 1
            else:
                span_end = True
        spans.append(k)
    return spans

def stack_stock_span(quotes):
    spans = [1]
    s = []
    s.append(0)
    for i in range(1, len(quotes)):
        while len(s) != 0 and quotes[s[-1]] <= quotes[i]:
            s.pop()
        if len(s) == 0:
            spans.append(i+1)
        else:
            spans.append(i - s[-1])
        s.append(i)
    return spans
    

def read_quotes(filename):
    quotes = []
    with open(filename) as quotes_file:
        for line in quotes_file:
            if line.startswith('#'):
                continue
            parts = line.split(',')
            quotes.append(parts[-1])
    return quotes

quotes = read_quotes("djia.csv")

min_time_taken = 10**10

for i in range(10):
    time_start = time.time()
    simple_stock_span(quotes)
    time_end = time.time()
    time_diff = time_end - time_start
    if time_diff < min_time_taken:
        min_time_taken = time_diff
        
print("Time for simple algorithm:", min_time_taken, "seconds.")

min_time_taken = 10**10

for i in range(10):
    time_start = time.time()
    stack_stock_span(quotes)
    time_end = time.time()
    time_diff = time_end - time_start
    if time_diff < min_time_taken:
        min_time_taken = time_diff
        
print("Time for stack-based algorithm:", min_time_taken, "seconds.")

