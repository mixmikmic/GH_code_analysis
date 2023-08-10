import datetime
import requests
import csv

# The URL for the streaming data to be scraped
streamPath = 'http://207.251.86.229/nyc-links-cams/LinkSpeedQuery.txt'

# The output file for the script
# Does not need to be created first
outputFilePath = 'C:/Users/JeffM/Documents/Projects/Spark Streaming/Data/'

from threading import Timer

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

from time import sleep

def scrapeWeb():
    """
    This function will read data from the streaming path specified in the beginning.
    It will then write it to an array, and then print it to a text file.
    """
    
    # Scrape the website
    r = requests.get(streamPath, stream = False)  # Stream = false to generate a new array each time
    
    # Handles missing encodings
    if r.encoding is None:
        r.encoding = 'utf-8'
    
    # Array to be written to a file
    nycTraffic = []
    
    # Loads data from requests to array
    for line in r.iter_lines(decode_unicode = True):
        # Replacing , with ; for multiple coordinates in a single column
        updatedLine = line.replace('"', '').replace(',', ';')
        if len(updatedLine.split('\t')) == 13:  # Filters out rows with missing elements
            nycTraffic.append(updatedLine)
    
    # Filters out items missing rows
    nycTraffic = [x for x in nycTraffic if len(x) > 200]

    # Writes array to the text file
    outputFile = outputFilePath+'nycTraffic_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.txt'
    with open(outputFile, "w") as output:
        writer = csv.writer(output, lineterminator = '\n')
        writer.writerows([line.split('\t') for line in nycTraffic[1:]])
    
    # Time stamp of when the web was scraped
    print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    

print("Starting...")
# Set RepeatedTimer(n) for number of seconds between each run
# The first job will start on a delay equal to n
rt = RepeatedTimer(20, scrapeWeb) # Auto-starts, no need for rt.start()

# Ending
try:
    # Set sleep(n) for number of seconds for the job to run
    sleep(3600)  # One hour
finally:
    rt.stop()

