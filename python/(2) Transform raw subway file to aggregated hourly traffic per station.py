import pandas as pd
import datetime
from dateutil.parser import parse
from datetime import datetime

subway_week = pd.read_csv("../Resources/Data/Raw/mta-subway-volume/turnstile_160319_mod.csv")

# a) remove entries which aren't "Regular" (i.e. counts which are made for maintenance purposes and not on the hourly schedule)
subway_week = subway_week[subway_week["DESC"]=="REGULAR"]

# This leads to some missing indicies indicies (e.g. 14733) which is leading into problems down the road.
# ... so we will fix that
# subway_week[14730:14740]
subway_week.index = range(len(subway_week))

# Improvement Opportunity: Make date parsing faster via: http://ze.phyr.us/faster-strptime/
subway_week["DATE"] = [parse(date_string).date() for date_string in subway_week["DATE"]]
subway_week["TIME"] = [parse(date_string).time() for date_string in subway_week["TIME"]]

subway_week["DATE_TIME"] = [datetime.combine(subway_week["DATE"][i], subway_week["TIME"][i]) for i in range(len(subway_week))]

subway_week.to_pickle("../Resources/Pickles/subway_traffic.pickle")
#subway_week = pd.read_pickle("../Resources/Pickles/subway_traffic.pickle")





small = subway_week.head()
small

