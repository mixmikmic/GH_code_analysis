import findspark

# provide path to your spark directory directly
findspark.init("~/spark2")

import pyspark

import findspark

# provide path to your spark directory directly
findspark.init("/Users/soumendra/spark2/")

import pyspark

sc = pyspark.SparkContext(appName="helloworld")

# let's test our setup by counting the number of nonempty lines in a text file
lines = sc.textFile('/Users/soumendra/helloworld')
lines_nonempty = lines.filter( lambda x: len(x) > 0 )
lines_nonempty.count()

# let's test our setup by counting the number of nonempty lines in a text file
lines = sc.textFile('/Users/soumendra/helloworld')
lines_nonempty = lines.filter( lambda x: len(x) > 0 )
lines_nonempty.count()

get_ipython().run_cell_magic('bash', '', 'text="Python is a fun language,\\nbut then what language\\nis not, if\\nI may ask. But Python\\nis also."\n\necho -e $text > python.txt\ncat python.txt')

lines = sc.textFile("/Users/soumendra/python.txt")
pythonLines = lines.filter(lambda line: "Python" in line)
print("No of lines containing 'Python':", pythonLines.count())

def isprime(n):
    """
    check if integer n is a prime
    """
    # make sure n is a positive integer
    n = abs(int(n))
    # 0 and 1 are not primes
    if n < 2:
        return False
    # 2 is the only even prime number
    if n == 2:
        return True
    # all other even numbers are not primes
    if not n & 1:
        return False
    # range starts with 3 and only needs to go up the square root of n
    # for all odd numbers
    for x in range(3, int(n**0.5)+1, 2):
        if n % x == 0:
            return False
    return True

# Create an RDD of numbers from 0 to 1,000,000
nums = sc.parallelize(range(1000000))

# Compute the number of primes in the RDD
print(nums.filter(isprime).count())

import re
from operator import add

filein = sc.textFile('/Users/soumendra/helloworld')

print('number of lines in file: %s' % filein.count())

filein_nonempty = filein.filter( lambda x: len(x) > 0 )
print('number of non-empty lines in file: %s' % filein_nonempty.count()) 

chars = filein.map(lambda s: len(s)).reduce(add)
print('number of characters in file: %s' % chars)

words = filein.flatMap(lambda line: re.split('\W+', line.lower().strip()))
words = words.filter(lambda x: len(x) > 3)

words = words.map(lambda w: (w,1))
words = words.reduceByKey(add)
print('number of words with more than 3 characters in file: %s' % words.count())

## Spark Application Template - execute with spark-submit

## Imports
from pyspark import SparkConf, SparkContext

## Module Constants
APP_NAME = "Name of Application"  #helps in debugging

## Closure Functions

## Main functionality

def main(sc):
    pass

if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)

    # Execute Main functionality
    main(sc)

# To close or exit the program use sc.stop() or sys.exit(0)

from add import *

add(5,6)

import findspark

# provide path to your spark directory directly
findspark.init("/Users/soumendra/spark2/")

import pyspark

get_ipython().magic('matplotlib inline')
## Imports
import csv
import matplotlib.pyplot as plt

from io import StringIO
from datetime import datetime
from collections import namedtuple
from operator import add, itemgetter
from pyspark import SparkConf, SparkContext

## Module Constants
APP_NAME = "Flight Delay Analysis"
DATE_FMT = "%Y-%m-%d"
TIME_FMT = "%H%M"

fields   = ('date', 'airline', 'flightnum', 'origin', 'dest', 'dep',
            'dep_delay', 'arv', 'arv_delay', 'airtime', 'distance')
Flight   = namedtuple('Flight', fields)

## Closure Functions
def parse(row):
    """
    Parses a row and returns a named tuple.
    """

    row[0]  = datetime.strptime(row[0], DATE_FMT).date()
    row[5]  = datetime.strptime(row[5], TIME_FMT).time()
    row[6]  = float(row[6])
    row[7]  = datetime.strptime(row[7], TIME_FMT).time()
    row[8]  = float(row[8])
    row[9]  = float(row[9])
    row[10] = float(row[10])
    return Flight(*row[:11])

def split(line):
    """
    Operator function for splitting a line with csv module
    """
    reader = csv.reader(StringIO(line))
    return reader.next()

def plot(delays):
    """
    Show a bar chart of the total delay per airline
    """
    airlines = [d[0] for d in delays]
    minutes  = [d[1] for d in delays]
    index    = list(xrange(len(airlines)))

    fig, axe = plt.subplots()
    bars = axe.barh(index, minutes)

    # Add the total minutes to the right
    for idx, air, min in zip(index, airlines, minutes):
        if min > 0:
            bars[idx].set_color('#d9230f')
            axe.annotate(" %0.0f min" % min, xy=(min+1, idx+0.5), va='center')
        else:
            bars[idx].set_color('#469408')
            axe.annotate(" %0.0f min" % min, xy=(10, idx+0.5), va='center')

    # Set the ticks
    ticks = plt.yticks([idx+ 0.5 for idx in index], airlines)
    xt = plt.xticks()[0]
    plt.xticks(xt, [' '] * len(xt))

    # minimize chart junk
    plt.grid(axis = 'x', color ='white', linestyle='-')

    plt.title('Total Minutes Delayed per Airline')
    plt.show()

## Main functionality
def main(sc):

    # Load the airlines lookup dictionary
    airlines = dict(sc.textFile("ontime/airlines.csv").map(split).collect())

    # Broadcast the lookup dictionary to the cluster
    airline_lookup = sc.broadcast(airlines)

    # Read the CSV Data into an RDD
    flights = sc.textFile("ontime/flights.csv").map(split).map(parse)

    # Map the total delay to the airline (joined using the broadcast value)
    delays  = flights.map(lambda f: (airline_lookup.value[f.airline],
                                     add(f.dep_delay, f.arv_delay)))

    # Reduce the total delay for the month to the airline
    delays  = delays.reduceByKey(add).collect()
    delays  = sorted(delays, key=itemgetter(1))

    # Provide output from the driver
    for d in delays:
        print("%0.0f minutes delayed\t%s" % (d[1], d[0]))

    # Show a bar chart of the delays
    plot(delays)

if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setMaster("local[*]")
    conf = conf.setAppName(APP_NAME)
    sc   = SparkContext(conf=conf)
    # Uncomment the lines above when running the application with "submit" (spark-submit app.py)
    # Comment the lines above out when running in IPython Notebook

    # Execute Main functionality
    main(sc)

