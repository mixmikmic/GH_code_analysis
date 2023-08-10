import pandas as pd #more info at http://pandas.pydata.org/
import numpy as np #more info at http://www.numpy.org/
import matplotlib.pyplot as plt #some examples for you at http://matplotlib.org/gallery.html 
from matplotlib import gridspec #more info at http://matplotlib.org/api/gridspec_api.html
import seaborn as sns

plt.style.use('ggplot') #For improving the visualization style (options: grayscale, bmh, dark_background, ggplot, and fivethirtyeight)

#Magic line: to show the plots inline in the Jupyter Notebook
get_ipython().magic('matplotlib inline')

import warnings
warnings.filterwarnings('ignore') #To avoid showing annoying warns

# This operation is extremely timing consuming. At least you only need to do it at ONCE!
get_ipython().system("tshark -n -r 'ipstresser_booter.pcap' -E separator=\\;  -E header=y -T fields -e frame.time_epoch -e ip.proto -e ip.src -e ip.dst -e udp.srcport -e udp.dstport -e tcp.srcport -e tcp.dstport -e frame.len -e dns.qry.type -e dns.qry.name -e dns.resp.name > pcap.txt")

df = pd.read_csv('pcap.txt', error_bad_lines=False, sep=';')

def timestamp2datetime(series):
    return  pd.to_datetime(series,unit='s',errors='coerce')

def bytes2bits(series):
    try:
        return  series*8
    except AttributeError:
        return series     

df['frame.time_epoch']=timestamp2datetime(df['frame.time_epoch'])

df['frame.len']=bytes2bits(df['frame.len'])

df.head()

ip_dst=df['ip.dst'].value_counts()
# Showing only the top ones
ip_dst.head()

fig = plt.figure(figsize=(4,4))

ax = plt.subplot2grid((1,1), (0,0))
ip_dst.plot(ax=ax,kind='pie', autopct='%1.1f%%', startangle=270, fontsize=10,title="Target IP")
ax.set_ylabel("")

fig.show()

top1_target_ip=ip_dst.index[0]
top1_target_ip

target_network=top1_target_ip.split('.')[0]+'.'+top1_target_ip.split('.')[1]
target_network

ip_proto=df['ip.proto'].value_counts()
ip_proto.head()

fig = plt.figure(figsize=(4,4))

ax = plt.subplot2grid((1,1), (0,0))
ip_proto.plot(ax=ax,kind='pie', autopct='%1.1f%%', startangle=270, fontsize=10,title="IP Protocols (number)")
ax.set_ylabel("")

fig.show()

top1_target_ip_proto=ip_proto.index[0]
top1_target_ip_proto

udp_srcip_dstip_top1=df[df['ip.dst']==top1_target_ip]['ip.src'].value_counts()
udp_srcport_top1=df[df['ip.dst']==top1_target_ip]['udp.srcport'].value_counts()
udp_dstport_top1=df[df['ip.dst']==top1_target_ip]['udp.dstport'].value_counts()

# udp_srcip_dstip_top2=df[df['ip.dst']==top2_target_ip]['ip.src'].value_counts()
# udp_srcport_top2=df[df['ip.dst']==top2_target_ip]['udp.srcport'].value_counts()
# udp_dstport_top2=df[df['ip.dst']==top2_target_ip]['udp.dstport'].value_counts()

fig = plt.figure(figsize=(12,6))
fig.subplots_adjust(wspace=1)

ax = plt.subplot2grid((2,3), (0,0))
udp_srcip_dstip_top1.plot(kind='pie',ax=ax, autopct='%1.1f%%', startangle=270, fontsize=10,title="Source IPs targetting "+top1_target_ip)
ax.set_ylabel("")

ax1 = plt.subplot2grid((2,3), (0,1))
udp_srcport_top1.plot(kind='pie',ax=ax1, autopct='%1.1f%%', startangle=270, fontsize=10,title="Source UDP Ports Distribution")
ax1.set_ylabel("")

ax2 = plt.subplot2grid((2,3), (0,2))
udp_dstport_top1.plot(kind='pie',ax=ax2, autopct='%1.1f%%', startangle=270, fontsize=10,title="Destination UDP Ports Distribution")
ax2.set_ylabel("")

# ax3 = plt.subplot2grid((2,3), (1,0))
# udp_srcip_dstip_top2.plot(kind='pie',ax=ax3, autopct='%1.1f%%', startangle=270, fontsize=10,title="Source IPs targetting "+top2_target_ip)
# ax3.set_ylabel("")

# ax4 = plt.subplot2grid((2,3), (1,1))
# udp_srcport_top2.plot(kind='pie',ax=ax4, autopct='%1.1f%%', startangle=270, fontsize=10,title="Source UDP Ports Distribution")
# ax4.set_ylabel("")

# ax5 = plt.subplot2grid((2,3), (1,2))
# udp_dstport_top2.plot(kind='pie',ax=ax5, autopct='%1.1f%%', startangle=270, fontsize=10,title="Destination UDP Ports Distribution")
# ax5.set_ylabel("")

fig.show()

occurrence_dnsquery=df['dns.qry.name'].value_counts()
occurrence_dnsquery.head()

fig = plt.figure(figsize=(4,4))
ax = plt.subplot2grid((1,1), (0,0))
occurrence_dnsquery.plot(kind='barh',ax=ax, fontsize=10, title="DNS query")
ax.set_ylabel("")

fig.show()

top_dnsquery=occurrence_dnsquery.index[0]
top_dnsquery

max(df['frame.len'][df['dns.qry.name']==top_dnsquery])

# MANUAL SELECTION: Considering the attack from ONE to ONE port!!!
attack_records=df[df['ip.dst']==top1_target_ip]                [df['ip.src'].str.contains(target_network)==False]                [df['ip.proto']==top1_target_ip_proto]                [df['udp.srcport']==udp_srcport_top1.index[0]]                [df['udp.dstport']==udp_dstport_top1.index[0]]

# # MANUAL SELECTION: Considering the attack from ONE to MANY ports!!!
# attack_records=df[df['ip.dst']==top1_target_ip]\
#                 [df['ip.src'].str.contains(target_network)==False]\
#                 [df['ip.proto']==top1_target_ip_proto]\
#                 [df['udp.srcport']==udp_srcport_top1.index[0]]\
#                 [df['dns.qry.name'].str.contains(top_dnsquery)==True]
                
# # MANUAL SELECTION: Considering the attack from MANY to ONE port!!!
# attack_records=df[df['ip.dst']==top1_target_ip]\
#                 [df['ip.src'].str.contains(target_network)==False]\
#                 [df['ip.proto']==top1_target_ip_proto]\
#                 [df['udp.dstport']==udp_dstport_top1.index[0]]\
#                 [df['dns.qry.name'].str.contains(top_dnsquery)==True]

# # MANUAL SELECTION: Considering the attack from MANY to MANY portS!!!
# attack_records=df[df['ip.dst']==top1_target_ip]\
#                 [df['ip.src'].str.contains(target_network)==False]\
#                 [df['ip.proto']==top1_target_ip_proto]\
#                 [df['udp.srcport']==udp_srcport_top1.index[0]]\
#                 [df['udp.dstport']==udp_dstport_top1.index[0]]\
#                 [df['dns.qry.name'].str.contains(top_dnsquery)==True]

remaining_records=df[~df.isin(attack_records)]

overall_bps=df.set_index(['frame.time_epoch']).groupby(pd.TimeGrouper(freq='S')).agg(['sum'])['frame.len']
attack_bps=attack_records.set_index(['frame.time_epoch']).groupby(pd.TimeGrouper(freq='S')).agg(['sum'])['frame.len']
attack_bps_median=attack_bps.median()
attack_bps_peak=max(attack_bps['sum'])

remaining_bps=remaining_records.set_index(['frame.time_epoch']).groupby(pd.TimeGrouper(freq='S')).agg(['sum'])['frame.len']

fig = plt.figure(figsize=(12,4))

ax = plt.subplot2grid((1,1), (0,0))

attack_bps.plot(ax=ax, lw=1)
ax.fill_between(attack_bps.index, 0, attack_bps['sum'],color='r')

ax.annotate(str(attack_bps_peak/10e6)+' Mb/s [peak]', (str(attack_bps[attack_bps['sum'] == attack_bps_peak].index.values[0]), attack_bps_peak),
             xytext=(0, 0), textcoords='offset points')
ax.annotate(str(attack_bps_median[0]/10e6)+' Mb/s [median]', (str(attack_bps.index.values[0]), attack_bps_median),
             xytext=(90, 0), textcoords='offset points')

remaining_bps.plot(ax=ax)

ax.legend(['Attacks records','Remaining records'])
ax.set_ylabel("Data [bit]")
ax.set_xlabel("Time [second]")

fig.show()

print 'Records:', len(df),'records (100%),',len(attack_records),'attack records (',len(attack_records)*100/len(df),'%)',',', len(df)-len(attack_records),'remaining records (',(len(df)-len(attack_records))*100/len(df),'% )'

print 'Trace duration:',max(df['frame.time_epoch'])-min(df['frame.time_epoch'])
print 'Attack duration:',max(attack_records['frame.time_epoch'])-min(attack_records['frame.time_epoch'])

len(attack_records['ip.src'].unique())

top5_srcips_pkts=attack_records['ip.src'].value_counts().head(5).sort_values()
top5_srcips_pkts

fig = plt.figure()
ax = plt.subplot2grid((1,1), (0,0))
top5_srcips_pkts.plot(kind='barh')
ax.set_ylabel("Source IP")
ax.set_xlabel("Packets")

fig.show()

top5_srcips_bits=attack_records.groupby('ip.src').agg(['sum'])['frame.len'].sort('sum',ascending=False).head(5).sort_values('sum')
top5_srcips_bits

fig = plt.figure()
ax = plt.subplot2grid((1,1), (0,0))
top5_srcips_bits.plot(kind='barh',ax=ax,legend=False)
ax.set_ylabel("Source IP")
ax.set_xlabel("Bits")

fig.show()

top1_srcips_bits = top5_srcips_bits.index[0]
top1_srcips_bits 

import subprocess
p = subprocess.Popen(['whois', '-h', 'whois.cymru.com','\"', '-v',top1_srcips_bits,'\"' ], stdout=subprocess.PIPE)

print p.communicate()[0]

