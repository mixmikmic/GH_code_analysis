from pyflightdata import *

get_countries()[:5]

get_airlines()[:5]

get_airports('India')[10:15]

#pass the airline-code from get_airlines
get_fleet('emirates-ek-uae')

#pass airline-code from get_airlines to see all current live flights
get_flights('emirates-ek-uae')[:10]

get_history_by_flight_number('AI101')[-5:]

get_history_by_tail_number('9V-SMA')[-5:]

get_info_by_tail_number('9V-SMA')

