# Some helpers

import requests

def _request(url, params=None):
    """
    Wraps web request
    """
    base_url = 'https://api.ncbi.nlm.nih.gov/variation/v0/'
    r = requests.get(base_url + url, params=params)
    if r.status_code != 200:
        raise Exception(r.json()['error'])
    return r.json()['data']

def _parse_spdi(data):
    """
    Parse spdi json into string
    """
    return "%s:%d:%s:%s" % (data['seq_id'], data['position'],  data['deleted_sequence'], data['inserted_sequence'])

def contextual(spdi):
    """
    For a given SPDI gets its contextual representation
    """
    data = _request("spdi/" + spdi + "/contextual")
    return _parse_spdi(data)

def canonical(spdi):
    """
    For a given contextual SPDI find its canonical representation on most recent annotation run
    """
    data = _request("spdi/" + spdi + "/canonical_representative")
    return _parse_spdi(data)

def get_rsid(spdi):
    """
    Try to find rsid for a given canonical SPDI
    """
    data = _request("spdi/" + spdi + "/rsids")
    return data['rsids']

def spdi_to_hgvs(spdi):
    """
    Convert SPDI to HGVS
    """
    data = _request("spdi/" + spdi + "/hgvs")
    return data['hgvs']

def hgvs_to_contextual(hgvs, assembly = "GCF_000001405.25"):
    """
    For a given HGVS get a list of SPDIs.
    """
    data = _request("hgvs/" + hgvs + "/contextuals", {"assembly": assembly })
    return [_parse_spdi(spdi) for spdi in data['spdis']]

def get_rs_data(rsid):
    """
    Retrieve RS data in json for a given rsid
    """
    r = requests.get('https://api.ncbi.nlm.nih.gov/variation/v0/beta/refsnp/' + str(rsid))
    if r.status_code != 200:
        raise Exception('Request failed')
    return r.json()

contextual('NC_000001.10:12345::C')

canonical(contextual('NC_000001.10:12345::C'))

get_rsid('NC_000008.11:19956017:1:G')

get_rs_data(268)

spdi_to_hgvs('NC_000008.11:19956017:1:G')

hgvs_to_contextual('NC_000008.11:g.19956018A>G')

