import pandas as pd
import matplotlib.pyplot as plt
import dateutil.parser as dp
import requests
import json
import os.path

from collections import namedtuple

DATA_CACHE_PATH="data_cache"

get_ipython().system(' curl http://ftp.ripe.net/ripe/atlas/probes/archive/meta-latest | bunzip2 > data_cache/all_probes.json')

AtlasProbe = namedtuple("AtlasProbe",
           ("pid", "version", "nat", "ip4", "ip6", "asn4", "asn6", "cc", "lon", "lat"))

def extract_atlas_probe(pobj):
    
    if "address_v4" in pobj and pobj['address_v4'] is not None:
        ip4 = pobj["address_v4"]
    elif "prefix_v4" in pobj:
        ip4 = pobj["prefix_v4"]
    else:
        ip4 = None

    if "address_v6" in pobj and pobj['address_v6'] is not None:
        ip6 = pobj["address_v6"]
    elif "prefix_v4" in pobj:
        ip6 = pobj["prefix_v6"]
    else:
        ip6 = None

    if "asn_v4" in pobj:
        asn4 = pobj["asn_v4"]
    else:
        asn4 = None

    if "asn_v6" in pobj:
        asn6 = pobj["asn_v6"]
    else:
        asn6 = None

    if "tags" in pobj:
        if len(pobj['tags']) > 0 and isinstance(pobj['tags'][0], dict):
            alltags = [tag['slug'] for tag in pobj['tags']]
        else:
            alltags = pobj['tags']

        if "system-v1" in alltags:
            version = 1
        elif "system-v2" in alltags:
            version = 2
        elif "system-v3" in alltags:
            version = 3
        elif "system-anchor" in alltags:
            version = 4
        else:
            version = 0

        nat = "nat" in alltags
    else:
        version = None
        nat = None
        
    # Short circuit: never connected means don't load
    if "status" in pobj and pobj['status'] == 0:
        version = 0
    
    if "geometry" in pobj and "coordinates" in pobj['geometry']:
        (lon, lat) = pobj['geometry']['coordinates']
    elif "longitude" in pobj and "latitude" in pobj:
        lon = pobj['longitude']
        lat = pobj['latitude']
    else:
        lon = None
        lat = None

    return AtlasProbe(pobj["id"], version, nat, ip4, ip6, asn4, asn6,
                      pobj["country_code"], lon, lat)



def probe_dataframe_from_file(filename):
    data = []
    
    # make a giant array
    with open(filename) as stream:
        all_probes = json.loads(stream.read())
        for pobj in all_probes["objects"]:
            data.append(extract_atlas_probe(pobj))

    # create a dataframe from it
    df = pd.DataFrame(data, columns=AtlasProbe._fields)
    
    # indexed by probe ID
    df.index = df['pid']
    del(df['pid'])
    
    # and return it
    return df

probe_df = probe_dataframe_from_file("data_cache/all_probes.json")

AtlasAnchor = namedtuple("AtlasAnchor",
           ("aid", "name", "pid", "ip4", "ip6", "asn4", "asn6", "cc", "lon", "lat"))

def extract_atlas_anchor(aobj):
    
    if "id" in aobj:
        aid = int(aobj["id"])
    else:
        aid = None

    if "fqdn" in aobj:
        name = aobj["fqdn"]
    else:
        name = None
    
    if name is not None and name.endswith(".anchors.atlas.ripe.net"):
        name = name[:-23]
        
    if "probe" in aobj:
        pid = int(aobj["probe"])
    else:
        pid = None
    
    if "ip_v4" in aobj:
        ip4 = aobj["ip_v4"]
    else:
        ip4 = None
    
    if "ip_v6" in aobj:
        ip6 = aobj["ip_v6"]
    else:
        ip6 = None

    if "as_v4" in aobj and aobj['as_v4'] is not None:
        asn4 = int(aobj["as_v4"])
    else:
        asn4 = None

    if "as_v6" in aobj and aobj['as_v6'] is not None:
        asn6 = int(aobj["as_v6"])
    else:
        asn6 = None
        
    if "country" in aobj:
        cc = aobj['country']
    else:
        cc = None
    
    if "geometry" in aobj and "coordinates" in aobj['geometry']:
        (lon, lat) = aobj['geometry']['coordinates']
    elif "longitude" in aobj and "latitude" in aobj:
        lon = aobj['longitude']
        lat = aobj['latitude']
    else:
        lon = None
        lat = None

    return AtlasAnchor(aid, name, pid, ip4, ip6, asn4, asn6, cc, lon, lat)

def anchor_dataframe_from_v2api():
    data = []
    url = "https://atlas.ripe.net/api/v2/anchors/"

    # iterate over API pagination
    while url is not None:
        res = requests.get(url)
        if not res.ok:
            raise RuntimeError("Atlas probe API request failed: "+repr(res.json()))

        api_content = json.loads(res.content.decode("utf-8"))
        url = api_content['next']
        for aobj in api_content["results"]:
            data.append(extract_atlas_anchor(aobj))
            
    # create a dataframe from it
    df = pd.DataFrame(data, columns=AtlasAnchor._fields)
    
    # indexed by probe ID
    df.index = df['aid']
    del(df['aid'])
    
    # and return it
    return df

anchor_df = anchor_dataframe_from_v2api()

AnchoringMetadata = namedtuple("AnchoringMetadata", ("aid", "type", "msm", "af", "proto", "start", "stop", "probe_ct"))

def anchoring_measurements_from_v2api(how_many = None):
    data = []
    url = "https://atlas.ripe.net/api/v2/anchor-measurements/?include=measurement"

    # iterate over API pagination
    while url is not None:
        res = requests.get(url)
        if not res.ok:
            raise RuntimeError("Atlas probe API request failed: "+repr(res.json()))

        api_content = json.loads(res.content.decode("utf-8"))
        url = api_content['next']
        for mobj in api_content["results"]:
            try:
                aid = int(mobj["target"].strip("/").split("/")[-1])
                typ = mobj["type"]
                msm = int(mobj["measurement"]["id"])
                af = int(mobj["measurement"]["af"])
                if "protocol" in mobj["measurement"]:
                    proto = mobj["measurement"]["protocol"]
                elif typ == "ping":
                    proto = "ICMP"
                else:
                    proto = None
                start = mobj["measurement"]["start_time"]
                stop = mobj["measurement"]["stop_time"]
                probe_ct = mobj["measurement"]["participant_count"]
            except Exception:
                continue
                
            data.append(AnchoringMetadata(aid, typ, msm, af, proto, start, stop, probe_ct))
        
        if how_many is not None and len(data) >= how_many:
            break
            
    # create a dataframe from it
    df = pd.DataFrame(data, columns=AnchoringMetadata._fields)
    
    # indexed by MSM ID
    df.index = df['msm']
    del(df['msm'])
    
    # and return it
    return df

ANCHOR_NAMES_WE_LIKE = [
    "ar-bue-as4270",   # Buenos Aires, Argentina
    "at-vie-as1120",   # Vienna, Austria
    "au-mel-as38796",  # Melbourne, Austraila
    "bd-dac-as24122",  # Dacca, Bangladesh
    "bg-sof-as8866",   # Sofia, Bulgaria
    "ca-mtr-as852",    # Montreal, Canada
    "ch-zrh-as559",    # Zurich, Switzerland
    "de-fra-as8763",   # Frankfurt, Germany
    "de-ham-as201709", # Hamburg, Germany
    "de-muc-as5539",   # Munich, Germany
    "ee-tll-as51349",  # Talinn, Estonia
    "es-bcn-as13041",  # Barcelona, Spain
    "fr-par-as1307",   # Paris, France
    "gr-ath-as5408",   # Athens, Greece
    "hk-hkg-as43996",  # Hong Kong SAR, China
    "hu-bud-as12303",  # Budapest, Hungary
    "id-jkt-as10208",  # Jakarta, Indonesia
    "ie-dub-as1213",   # Dublin, Ireland
    "in-bom-as33480",  # Mumbai, India
    "it-trn-as12779",  # Turin, Italy
    "jp-tyo-as2500",   # Tokyo, Japan
    "kz-ala-as21299",  # Almaty, Kazakhstan
    "nl-ams-as3333",   # Amsterdam, Holland
    "nz-wlg-as9834",   # Wellington, New Zealand
    "qa-doh-as8781",   # Doha, Qatar
    "ru-mow-as15835",  # Moscow, Russia
    "se-sto-as8674",   # Stockholm, Sweden
    "uk-lon-as5607",   # London, England
    "us-dal-as2914",   # Dallas, USA
    "us-den-as7922",   # Denver, USA
    "us-mia-as33280",  # Miami, USA
    "us-sjc-as22300",  # San Jose, USA
]

aid_by_name = anchor_df.loc[:,('name',)]
aid_by_name['aid'] = aid_by_name.index
aid_by_name.index = aid_by_name['name']
del aid_by_name['name']

ANCHORS_WE_LIKE = [aid_by_name.loc[aname]['aid'] for aname in ANCHOR_NAMES_WE_LIKE]

am_df = anchoring_measurements_from_v2api()
am2a_df = am_df[am_df['aid'].isin(ANCHORS_WE_LIKE)]
ping2a_df = am2a_df[am2a_df['type'] == 'ping']

MSMS_WE_LIKE = ping2a_df.index.values

START_TIME = int(dp.parse("2017-07-11T12:00:00Z").timestamp())
STOP_TIME = int(dp.parse("2017-07-11T18:00:00Z").timestamp())

# quick checkpoint here
MSMS_WE_LIKE = [1026364, 1026366, 1026392, 1026394, 1026400, 1026402, 1042256,
       1042258, 1043287, 1043289, 1404334, 1404336, 1423187, 1423189,
       1437285, 1437287, 1446418, 1446420, 1583041, 1583043, 1589863,
       1589865, 1591157, 1591159, 1664874, 1664876, 1665837, 1665839,
       1668852, 1668854, 1768006, 1768008, 1769992, 1769994, 1790233,
       1790235, 1849606, 1849608, 1990234, 1990236, 2055769, 2055771,
       2096535, 2096537, 2395061, 2395063, 2398551, 2398553, 2417651,
       2417653, 3295750, 3295764, 3315654, 3315657, 3614642, 3614645,
       3622419, 3622422, 6969365, 6969368, 7861647, 7861650, 8434916,
       8434919, 9180599, 9180602, 9180619, 9180622, 9180653, 9180656,
       9180667, 9180670, 9180705, 9180708, 9180725, 9180728, 9180767,
       9180770, 9180929, 9180932, 9180955, 9180958, 9180990, 9180994,
       9181094, 9181097, 9181100, 9181103, 9181175, 9181178, 9181254,
       9181257, 9181266, 9181269, 9181276, 9181279, 9181282, 9181285,
       9181297, 9181300, 9181317, 9181320, 9181352, 9181355, 9181364,
       9181367, 9181397, 9181400, 9181529, 9181532, 9181693, 9181696,
       9181746, 9181749, 9181778, 9181781, 9181821, 9181824, 9183385,
       9183388, 9183524, 9183527, 9183541, 9183544, 9183659, 9183662,
       9183787, 9183790]

Alp = namedtuple("Alp", ("time","af","proto","pid","sip","dip","rtt"))

RTT_NONE = 0.0
DATA_CACHE_PATH = 'data_cache'

def gen_dict(msm_ary):
    for a_res in msm_ary:
        yield a_res

def gen_alp(msm_ary):
    for a_res in msm_ary:
        if a_res['type'] == 'ping':
            if "rcvd" in a_res:
                for x in a_res["result"]:
                    rtt = None
                    try: 
                        rtt = float(x)
                    except:
                        try:
                            rtt = float(x['rtt'])
                        except:
                            pass
                    if rtt:
                        yield Alp(int(a_res['timestamp']), a_res['af'], a_res['proto'], 
                                  a_res['prb_id'], a_res['src_addr'], a_res['dst_addr'], 
                                  int(rtt * 1000))
        
        elif a_res['type'] == 'traceroute':
            if ('result' in a_res) and ('result' in a_res['result'][-1]):
                for h_res in a_res['result'][-1]['result']:
                    if ('from' in h_res) and ('rtt' in h_res) and (h_res['from'] == a_res['dst_addr']):
                        yield Alp(int(a_res['timestamp']), a_res['af'], a_res['proto'] + '_TR', 
                                  a_res['prb_id'], a_res['src_addr'], a_res['dst_addr'], h_res['rtt'])

        # For HTTP, return each subresult as a separate RTT sample
        elif a_res['type'] == 'http':
            for r_res in a_res['result']:
                if ('res' in r_res) and (r_res['res'] < 400):
                    yield Alp(a_res['timestamp'], r_res['af'], 'HTTP', 
                              a_res['prb_id'], r_res['src_addr'], r_res['dst_addr'], r_res['rt'])                    

        

def gen_msm(msm, gen=gen_alp, cachedir=None, start=None, stop=None):
    """
    Given an MSM, fetch it from the cache or from the RIPE Atlas API.
    Yield each separate result according to the generation function.
    """
    url = "https://atlas.ripe.net/api/v2/measurements/%u/results/" % (msm,)

    params = {"format": "json"}
    if start is not None and stop is not None:
        params["start"] = str(start)
        params["stop"] = str(stop)
    
    if cachedir and os.path.isdir(cachedir):
        filepath = os.path.join(cachedir, "measurement", "%u.json" % (msm,))

        # download if not present
        if not os.path.isfile(filepath):
            with open(filepath, mode="wb") as file:
                print("Cache miss, retrieving "+url)
                res = requests.get(url, params=params)

                if not res.ok:
                    raise "Atlas measurement API request failed: "+repr(res.json())
                
                file.write(res.content)

        # then read from cache
        with open(filepath) as stream:
            yield from gen(json.loads(stream.read()))

    else:
        # just read from the net
        res = requests.get(url, params=params)
        yield from gen(json.loads(res.content.decode("utf-8")))

def msm_dfgen(msms, cachedir=None, start=None, stop=None):
    for msm in msms:
        yield from gen_msm(msm, cachedir=cachedir, start=start, stop=stop)
              

df = pd.DataFrame(msm_dfgen(MSMS_WE_LIKE, cachedir=DATA_CACHE_PATH, start=START_TIME, stop=STOP_TIME))

aid_by_ip4 = anchor_df.loc[:,('ip4',)]
aid_by_ip4['aid'] = aid_by_ip4.index
aid_by_ip4.index = aid_by_ip4['ip4']
del aid_by_ip4['ip4']

aid_by_ip6 = anchor_df.loc[:,('ip6',)]
aid_by_ip6['aid'] = aid_by_ip6.index
aid_by_ip6.index = aid_by_ip6['ip6']
del aid_by_ip6['ip6']

df = pd.concat((df[df['af']==4].join(aid_by_ip4, on="dip"), df[df['af']==6].join(aid_by_ip6, on="dip"))).dropna()

df['time'] = pd.to_datetime(df['time'] * 1e9)
df['aid'] = pd.to_numeric(df['aid'], downcast='unsigned')

with pd.HDFStore('rtt.h5') as store:
    store['anchor_df'] = anchor_df
    store['probe_df'] = probe_df
    store['rtt_df'] = df



