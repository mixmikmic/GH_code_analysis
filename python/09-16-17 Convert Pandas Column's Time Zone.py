import pandas as pd
from pytz import all_timezones

#Show ten time zones
all_timezones[0:10]

#Create ten dates
dates = pd.Series(pd.date_range('2/2/2002', periods=10, freq='M'))
dates

#Adding Time Zones of pandas Series
dates_with_abidjan_time_zones = dates.dt.tz_localize('Africa/Abidjan')
dates_with_abidjan_time_zones

# Convert time zone
dates_with_london_time_zone = dates_with_abidjan_time_zones.dt.tz_convert('Europe/London')
dates_with_london_time_zone

