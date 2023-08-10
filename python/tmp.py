# this is just to silence 
get_ipython().magic('xmode plain')

from sqlitedict import SqliteDict

def harness(key, value):
    """ this tests what can be assigned in SqliteDict's keys and values """
    mydict = SqliteDict(":memory:")
    mydict[key] = value

from battle_tested import fuzz, success_map, crash_map

fuzz(harness, keep_testing=True) # keep testing allows us to collect "all" crashes

crash_map()

success_map()



