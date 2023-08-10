import sys, os
import requests

baseurl = 'http://api.busturnaround.nyc/api/v1/'

def queryBusTurnaroundAPI(endpoint, kv_dict): 
    '''
    Queries bus turnaround api according to endpoint and key_value dictionary 
    
    Possible enpoints & keys available here: http://api.busturnaround.nyc/
    
    Note: Data only available Aug 2014-Feb 2016
    
    Note: EWT calculation is not the correct one! 
    '''
    query = endpoint + '?'
    for k,v in kv_dict.iteritems():
        query = query + k + '=' + str(v) + '&'
    query = query[:-1]
    try: 
        r = requests.get((baseurl + query).replace('%2C', ''))
        print r.url
        return r.json()
    except: 
        print 'error'
        

kv_dict = {'months': '2016-01-01,2016-02-01', 'weekend': '0'}

x = queryBusTurnaroundAPI('wtp', kv_dict)

x = queryBusTurnaround('')





