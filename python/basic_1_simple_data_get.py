import clarindspace
import urllib
import csv
import os
import json
import logging
from pprint import pformat
from __future__ import print_function

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

def plot(arr):
    """
        Just testing one way of visualisation.
    """
    import pandas as pd
    from matplotlib import pyplot as plt
    from math import radians

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.scatter(x=[radians(float(deg)) for _1, deg in arr], y=[1] * len(arr))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    plt.show()

# PID to a clarin-dspace repository
# - metadata attached to the PID at http://hdl.handle.net/11346/TEST--HGGA?noredirect
pid_url = "http://hdl.handle.net/11346/TEST--HGGA"

# get urls to all bitstreams 
show_n = 10
for bitstream_mimetype, bitstream_url in clarindspace.item.bitstream_info_from_pid(pid_url, mimetype="text/csv"):
    print("Fetching [%s]" % bitstream_url)
    data_csv = csv.reader(urllib.urlopen(bitstream_url))
    data_csv = [ [x.strip() for x in line] for line in data_csv ]
    print("Number of rows (with header): %8d" % len(data_csv))
    for i, row in enumerate(data_csv):
        print(row)
        if i > show_n:
            break

plot(data_csv[1:3000])

