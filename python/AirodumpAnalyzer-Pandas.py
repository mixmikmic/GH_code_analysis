get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np


import netaddr 


import seaborn as sns
sns.set_color_codes(palette='deep')

airodump_csv = open('airodump.csv', 'r').read()

client_header = 'Station MAC, First time seen, Last time seen, Power, # packets, BSSID, Probed ESSIDs'

hdi = airodump_csv.index(client_header)

from StringIO import StringIO

ap_csv = StringIO(airodump_csv[:hdi])

client_csv = StringIO(airodump_csv[hdi:])

ap_df = pd.read_csv(ap_csv, 
                   sep=',', 
                   skipinitialspace=True,
                    parse_dates = ['First time seen', 'Last time seen']
                   )

client_df = pd.read_csv(client_csv,
                        sep=', ',
                        skipinitialspace=True,
                        engine='python',
                        parse_dates = ['First time seen', 'Last time seen']
                       )

ap_df.head(1)

client_df.head(1)

ap_df.columns

ap_df.rename(columns={
       'BSSID' : 'bssid',
       'First time seen' : 'firstseen',
       'Last time seen' : 'lastseen',
       'channel' : 'channel',
       'Speed' : 'speed',
       'Privacy' : 'privacy',
       'Cipher' : 'cipher',
       'Authentication' : 'authentication',
       'Power' : 'dbpower',
       '# beacons' : 'beacons',
       '# IV' : 'iv',
       'LAN IP' : 'ip',
       'ID-length' : 'idlen',
       'ESSID' : 'essid',
       'Key' : 'key'
   }, inplace=True)

ap_df.head(3)

set(ap_df.essid)

# Find all ESSIDs which is null i.e. Hidden SSID

ap_df[ap_df.essid.isnull()]

# Let's replace the NaNs with "Hidden SSID" 

ap_df.essid.fillna('Hidden SSID', inplace=True)

ap_df.essid.hasnans

ap_df[ap_df.essid == 'Hidden SSID'].head(3)

# Let us now get the ESSID counts

essid_stats = ap_df.essid.value_counts()

essid_stats

essid_stats.plot(kind='barh', figsize=(10,5))

ap_df.channel.value_counts()

ap_df.channel.value_counts().plot(kind='bar')

# AP vendors can be figured out by the first 3 bytes of the MAC address. 


manufacturer = ap_df.bssid.str.extract('(..:..:..)', expand=False)

manufacturer.head(10)

manufacturer.value_counts()

# https://pypi.python.org/pypi/netaddr

import netaddr

netaddr.OUI('10:8C:CF'.replace(':', '-')).registration().org

for x in manufacturer.value_counts().index[:10]: 
    print x

def manufac(oui) :
    try:
        return netaddr.OUI(oui.replace(':', '-')).registration().org
    except:
        return "Unknown"

[ manufac(oui) for oui in manufacturer.value_counts().index]

client_df.head(1)

client_df.columns = ['clientmac', 'firstseen', 'lastseen', 'power', 'numpkts', 'bssid', 'probedssids']

client_df.head(2)

client_df.bssid.head(10)

client_df['bssid'] = client_df.bssid.str.replace(',', '')

client_df.bssid.head(10)

all_probed_ssids_list = []

def createprobedlist(x) :
    if x:
        all_probed_ssids_list.extend(x.strip().split(','))
        
client_df.probedssids.apply(createprobedlist)

all_probed_ssids_list

set(all_probed_ssids_list)

client_df.count()

client_df.bssid.str.contains('not associated').value_counts()

