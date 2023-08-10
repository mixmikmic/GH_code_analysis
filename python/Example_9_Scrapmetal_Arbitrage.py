# Standard imports
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import datetime
get_ipython().magic('matplotlib inline')
# EveKit imports
from evekit.reference import Client
from evekit.util import convert_raw_time

# We've collected results from our backtest into a CSV file with format:
#
# snapshot time, profit, gross, cost, type
#
# We'll start by reading this data into an array.
#
results_file = "single_day_scrap.csv"

opportunities = []
fin = open(results_file, 'r')
for line in fin:
    columns = line.strip().split(",")
    opportunities.append(dict(time=datetime.datetime.strptime(columns[0], "%Y-%m-%d %H:%M:%S"),
                             profit=float(columns[1]),
                             gross=float(columns[2]),
                             cost=float(columns[3]),
                             type=columns[4]))

# Our offline script exploits parallelism to find opportunities more quickly (but a day
# of data still takes four hours to analyze).  A side effect is that our list of opportunities
# is unsorted, so we'll first sort by opportunity time.
opportunities = sorted(opportunities, key=lambda x: x['time'])

# Now we're ready to clean the list, collapsing adjacent opportunities into
# their first occurrence.  We'll use the same function as before:
def clean_opportunities(opps):
    new_opps = []
    stamp_map = {}
    types = set([x['type'] for x in opps])
    # Flatten opportunites for each type
    for next_type in types:
        stamp_list = []
        last = None
        for i in [x['time'] for x in opps if x['type'] == next_type]:
            if last is None:
                # First opportunity
                stamp_list.append(i)
            elif i - last > datetime.timedelta(minutes=5):
                # Start of new run
                stamp_list.append(i)
            last = i
        stamp_map[next_type] = stamp_list
    # Rebuild opportunities by only selecting opportunities in
    # the flattened lists.
    for opp in opps:
        type = opp['type']
        if opp['time'] in stamp_map[type]:
            new_opps.append(opp)
    # Return the new opportunity list
    return new_opps

opportunities = clean_opportunities(opportunities)

# Let's take a look at what we have
def display_opportunities(opps):
    for opp in opps:
        profit = "{:15,.2f}".format(opp['profit'])
        margin = "{:8.2f}".format(opp['profit'] / opp['cost'] * 100)
        print("ArbOpp time=%s  profit=%s  return=%s%%  type=%s" % (str(opp['time']), profit, margin, opp['type']))
    print("Total opportunities: %d" % len(opps))

display_opportunities(opportunities)   

# Now let's generate a summary to see what the total opportunity looks like.
total_profit = np.sum([x['profit'] for x in opportunities])
total_cost = np.sum([x['cost'] for x in opportunities])
total_return = total_profit / total_cost
print("Total opportunity profit: %s ISK" % "{:,.2f}".format(total_profit))
print("Total opportunity retrun: %s%% ISK" % "{:,.2f}".format(total_return * 100))

