from Storms import *

#define an object
tc=Storm()

url = 'https://metoc.ndbc.noaa.gov/RSSFeeds-portlet/img/jtwc/jtwc.rss' # JTWC

url='http://www.nhc.noaa.gov/index-at.xml' #NOAA Atlantic

url='http://www.nhc.noaa.gov/index-ep.xml' #NOAA Pacific

tc.parse(url=url)

tc.data

tc.parse_atcf(filename='../test/matthew14l.2016092912.trak.hwrf.atcfunix')

tc.name

tc.basin

tc.date

tc.data

tc.toSI()



