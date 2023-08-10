import shutil
import tempfile
import os
import time
import json
from tornado.websocket import websocket_connect
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from functools import reduce
import pandas as pd
import numpy as np

import declarativewidgets as widgets

widgets.init()

get_ipython().run_cell_magic('html', '', '<link rel="import" href="urth_components/urth-viz-chart/urth-viz-chart.html" is="urth-core-import">\n\n<template is="urth-core-bind" channel="counts">\n    <urth-viz-chart type=\'bar\' datarows=\'[[counts.data]]\' columns=\'[[counts.columns]]\' rotatelabels=\'30\'></urth-viz-chart>\n</template>')

topic_filter = ''

def set_topic_filter(value):
    global topic_filter
    topic_filter = value

get_ipython().run_cell_magic('html', '', '<link rel="import" href="urth_components/paper-input/paper-input.html"\n    is="urth-core-import" package="PolymerElements/paper-input">\n    \n<template is="urth-core-bind" channel="filter" id="filter-input">\n    <urth-core-function auto\n        id="set_topic_filter"\n        ref="set_topic_filter"\n        arg-value="{{topic_filter}}">\n    </urth-core-function>\n        \n    <paper-input label="Filter" value="{{topic_filter}}"></paper-input>\n</template>')

get_ipython().run_cell_magic('html', '', '<link rel="import" href="urth_components/paper-card/paper-card.html"\n    is="urth-core-import" package="PolymerElements/paper-card">\n\n<style is="custom-style">\n    paper-card.meetups-card {\n        max-width: 400px;\n        width: 100%;\n        \n        --paper-card-header: {\n            height: 100px;\n            border-bottom: 1px solid #e8e8e8;\n        };\n\n        --paper-card-header-image: {\n            height: 80px;\n            width: 80px !important;\n            margin: 10px;\n            border-radius: 50px;\n            width: auto;\n            border: 10px solid white;\n            box-shadow: 0 0 1px 1px #e8e8e8;\n        };\n        \n        --paper-card-header-image-text: {\n            left: auto;\n            right: 0px;\n            width: calc(100% - 130px);\n            text-align: right;\n            text-overflow: ellipsis;\n            overflow: hidden;\n        };\n    }\n    \n    .meetups-card .card-content a {\n        display: block;\n        overflow: hidden;\n        text-overflow: ellipsis;\n        white-space: nowrap;\n    }\n</style>\n\n<template is="urth-core-bind" channel="meetups" id="meetup-card">\n    <paper-card\n            class="meetups-card"\n            heading="[[meetup.member.member_name]]"\n            image="[[meetup.member.photo]]">\n        <div class="card-content">\n            <p><a href="[[meetup.event.event_url]]" target="_blank">[[meetup.event.event_name]]</a></p>\n            <p>[[getPrettyTime(meetup.event.time)]]</p>\n        </div>\n    </paper-card>\n</template>\n\n<!-- see https://github.com/PolymerElements/iron-validator-behavior/blob/master/demo/index.html -->\n<script>\n    (function() {\n        var dateStringOptions = {weekday:\'long\', year:\'numeric\', month: \'long\', hour:\'2-digit\', minute:\'2-digit\', day:\'numeric\'};\n        var locale = navigator.language || navigator.browserLanguage || navigator.systemLanguage || navigator.userLanguage;\n\n        var scope = document.querySelector(\'template#meetup-card\');\n        scope.getPrettyTime = function(timestamp) {\n            var d = new Date(timestamp);\n            return d.toLocaleDateString(locale, dateStringOptions);\n        }\n    })();\n</script>')

get_ipython().run_cell_magic('html', '', '<link rel="import" href="urth_components/webgl-globe/webgl-globe.html"\n  is="urth-core-import" package="http://github.com/ibm-et/webgl-globe.git">\n\n<template is="urth-core-bind" channel="venues">\n    <webgl-globe data=[[venue_data]]></webgl-globe>\n</template>')

def create_streaming_context(checkpoint_dir, sample_rate):
    '''
    Creates a new SparkContext and SparkStreamingContext. Done in a function
    to allow repeated start/stop of the streaming. Returns the streaming
    context instance.
    
    :param checkpoint_dir: Directory to use to track Spark job state
    :param sample_rate: Stream sampling rate in seconds
    '''
    # create a local SparkContext to start using as many CPUs as we can
    sc = SparkContext('local[*]')
    
    # wrap it in a StreamingContext that collects from the stream
    ssc = StreamingContext(sc, sample_rate)

    # Setup a checkpoint directory to keep total counts over time.
    ssc.checkpoint(os.path.join(checkpoint_dir, '.checkpoint'))
    
    return ssc

def retain_event(event):
    '''
    Returns true if the user defined topic filter is blank or if at least one
    group topic in the event exactly matches the user topic filter string.
    '''
    global topic_filter
    if topic_filter.strip() == '':
        return True
    return any(topic['urlkey'] == topic_filter for topic in event['group']['group_topics'])

def get_events(ssc, queue, for_each):
    '''
    Parses the events from the queue. Retains only those events that have at
    least one topic exactly matching the current topic_filter. Sends event
    RDDs to the for_each function. Returns the event DStream.
    '''
    msgs = ssc.textFileStream(queue)
    
    # Each event is a JSON blob. Parse it. Filter it.
    events = (msgs.map(lambda json_str: json.loads(json_str))
                  .filter(lambda event: retain_event(event)))

    # Send event data to a widget channel. This will be covered below.
    events.foreachRDD(for_each)
    
    return events

def update_topic_counts(new_values, last_sum):
    '''
    Sums the number of times a topic has been seen in the current sampling
    window. Then adds that to the number of times the topic has been
    seen in the past. Returns the new sum.
    '''
    return sum(new_values) + (last_sum or 0)

def get_topics(events, for_each):
    '''
    Pulls group topics from meetup events. Counts each one once and updates
    the global topic counts seen since stream start. Sends topic count RDDs
    to the for_each function. Returns nothing new.
    '''
    # Extract the group topic url keys and "namespace" them with the current topic filter
    topics = (events
                .flatMap(lambda event: event['group']['group_topics'])
                .map(lambda topic: ((topic_filter if topic_filter else '*', topic['urlkey']), 1)))
    
    topic_counts = topics.updateStateByKey(update_topic_counts)

    # Send topic data to a widget channel. This will be covered below.
    topic_counts.foreachRDD(for_each)

def get_venues(events, for_each):
    '''
    Pulls venu metadata from meetup events if it exists. Sends venue 
    dictionaries RDDs to the for_each function. Returns nothing new.
    '''
    venues = (events
        .filter(lambda event: 'venue' in event)
        .map(lambda event: event['venue']))
    
    # Send topic data to a widget channel
    venues.foreachRDD(for_each)

from declarativewidgets import channel

def sample_event(rdd):
    '''
    Takes an RDD from the event DStream. Takes one event from the RDD.
    Substitutes a placeholder photo if the member who RSVPed does not
    have one. Publishes the event metadata on the meetup channel.
    '''
    event = rdd.take(1)
    if len(event) > 0:
        evt = event[0]
        
        # use a fallback photo for those members without one
        if 'photo' not in evt['member'] or evt['member']['photo'] is None:
            evt['member']['photo'] = 'http://photos4.meetupstatic.com/img/noPhoto_50.png'

        channel('meetups').set('meetup', evt)

def get_topic_counts(rdd):
    '''
    Takes an RDD from the topic DStream. Takes the top 25 topics by occurrence
    and publishes them in a pandas DataFrame on the counts channel.
    '''
    #counts = rdd.takeOrdered(25, key=lambda x: -x[1])
    filterStr = topic_filter if topic_filter else '*'
    counts = (rdd
                .filter(lambda x: x[0][0] == filterStr) # keep only those matching current filter
                .takeOrdered(25, key=lambda x: -x[1]))  # sort in descending order, taking top 25
    if not counts:
        # If there are no data, the bar chart will error out. Instead,
        # we send a tuple whose count is zero.
        counts = [('NO DATA', 0)]
    else:
        # Drop the topic filter from the tuple
        counts = list(map(lambda x: (x[0][1], x[1]), counts))
    df = pd.DataFrame(counts)
    channel('counts').set('counts', df)

venue_data = []
lon_bins = np.linspace(-180, 180, 361)
lat_bins = np.linspace(-90, 90, 181)
scale=100

def aggregate_venues(rdd):
    '''
    Takes an RDD from the venues DStream. Builds a histogram of events by 
    latitude and longitude. Publishes the histogram as a list of three-tuples
    on the venues channel.
    
    Note: To improve scalability, this binning should be performed
    on the Spark workers, not collected and performed on the driver.
    '''
    global venue_data

    # create new lists from previous data and new incoming venues
    venues = rdd.collect()
    lats = [v[0] for v in venue_data] + [x['lat'] for x in venues]
    lons = [v[1] for v in venue_data] + [x['lon'] for x in venues]
    weights = [v[2] for v in venue_data] + ([1./scale] * len(venues))
    
    # create histogram from aggregate data
    density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins], weights=weights)
    venue_data = [[lat-90, lon-180, min(mag,1)]
                     for lat,dlats in enumerate(density)
                     for lon,mag in enumerate(dlats)
                     if mag > 0]
    
    channel('venues').set('venue_data', venue_data)

class FileRingReceiver(object):
    '''
    Hack around lack of custom DStream receivers in Python: 
    Create a ring buffer of UTF-8 text files on disk.
    '''
    def __init__(self, max_batches=10):
        self.queue = tempfile.mkdtemp()
        self.batch_count = 0
        self.max_batches = max_batches
        
    def __del__(self):
        self.destroy()
        
    def put(self, text):
        # ignore sentinels
        if text is None: return
        with open(os.path.join(self.queue, str(self.batch_count)), 'w', encoding='utf-8') as f:
            f.write(text)
        if self.batch_count >= self.max_batches:
            oldest = str(self.batch_count - self.max_batches)
            os.remove(os.path.join(self.queue, str(oldest)))
        self.batch_count += 1
        
    def destroy(self):
        shutil.rmtree(self.queue, ignore_errors=True)

conn_future = None
ssc = None
receiver = None

def start_stream():
    '''
    Creates a websocket client that pumps events into a ring buffer queue. Creates
    a SparkStreamContext that reads from the queue. Creates the events, topics, and
    venues DStreams, setting the widget channel publishing functions to iterate over
    RDDs in each. Starts the stream processing.
    '''
    global conn_future
    global ssc
    global receiver
    
    receiver = FileRingReceiver(max_batches=100)  
    conn_future = websocket_connect('ws://stream.meetup.com/2/rsvps', on_message_callback=receiver.put)
    ssc = create_streaming_context(receiver.queue, 5)
    events = get_events(ssc, receiver.queue, sample_event)
    get_topics(events, get_topic_counts)
    get_venues(events, aggregate_venues)
    ssc.start()
    
def shutdown_stream():
    '''
    Shuts down the websocket, stops the streaming context, and cleans up the file ring.
    '''
    global conn_future
    global ssc
    global receiver
    
    conn_future.result().close()
    ssc.stop()
    receiver.destroy()

get_ipython().run_cell_magic('html', '', '<link rel="import" href="urth_components/paper-toggle-button/paper-toggle-button.html"\n    is="urth-core-import" package="PolymerElements/paper-toggle-button#v1.0.10">\n    \n<template is="urth-core-bind">\n    <urth-core-function id="streamFunc" ref="start_stream"></urth-core-function>\n    <urth-core-function id="shutdownFunc" ref="shutdown_stream"></urth-core-function>\n</template>\n\n<style is="custom-style">\n    paper-toggle-button {\n        --default-primary-color: green;\n    }\n    \n    paper-toggle-button:hover {\n        cursor: pointer;\n    }\n        \n    .toggle-btn-container {\n        margin: 1em 0;\n        text-align: right;\n    }\n    \n    #stream-label {\n        font-size: larger;\n        margin: 0;\n        padding: 0 0.5em;\n    }\n</style>\n\n<div class="toggle-btn-container">\n    <paper-toggle-button id="stream-btn"></paper-toggle-button>\n    <label id="stream-label">Stream</label>\n</div>\n\n<script>\n    $(\'#stream-btn\').on(\'change\', function() {\n        if ($(this).attr(\'checked\')) {\n            // start streaming\n            console.warn(\'Starting Spark Streaming\');\n            $(\'#streamFunc\').get(0).invoke();\n        } else {\n            // stop streaming\n            console.warn(\'Stopping Spark Streaming\');\n            $(\'#shutdownFunc\').get(0).invoke();\n        }\n    });\n</script>')

