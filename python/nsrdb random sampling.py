import numpy as np
import os
import pandas as pd

cwd = os.getcwd()

def get_gid(meta):
    """
    Extract random gid from nsrdb ensuring samples are randomly sampled from available states and counties
    
    Parameters
    ----------
    meta : 'pandas.DataFrame'
        DataFrame of meta data from which to randomly samples pixels

    Returns
    -------
    gid : 'int'
        Selected gid
    """
    if len(meta['state'].unique()) > 1:
        state = np.random.choice(meta['state'].unique(), 1)[0]
        meta = meta.loc[meta['state'] == state]

    if len(meta['county'].unique()) > 1:
        county = np.random.choice(meta['county'].unique(), 1)[0]
        meta = meta.loc[meta['county'] == county]
    
    gid = np.random.choice(meta['gid'].values, 1)[0]
    return gid


def sample_nsrdb(meta, samples):
    """
    Randomly sample from nsrdb meta data
    Samples are selected from available countries, states, or counties
    
    Parameters
    ----------
    meta : 'pandas.DataFrame'
        DataFrame of meta data from which to randomly samples pixels
    samples : 'int'
        Number of samples to select

    Returns
    -------
    'pandas.DataFrame'
        Meta data for selected pixels
    """
    gids = []
    if len(meta['country'].unique()) > 1:
        countries = np.random.choice(meta['country'].unique(), samples)
        for country in countries:
            country_meta = meta.loc[meta['country'] == country]
            gids.append(get_gid(country_meta))      
    elif len(meta['state'].unique()) > 1:
        states = np.random.choice(meta['state'].unique(), samples)
        for state in states:
            state_meta = meta.loc[meta['state'] == state]
            gids.append(get_gid(state_meta))
    elif len(meta['county'].unique()) > 1:
        counties = np.random.choice(meta['county'].unique(), samples)
        for county in counties:
            county_meta = meta.loc[meta['county'] == county]
            gids.append(get_gid(county_meta))
    else:
        gids = np.random.choice(meta['gid'], samples)
        
    return meta.loc[gids]

path = os.path.join(cwd, 'nsrdb_meta.csv')
meta = pd.read_csv(path)
meta['gid'] = np.arange(len(meta))

countries = sample_nsrdb(meta, 5)
countries

US = meta.loc[meta['country'] == 'United States']
states = sample_nsrdb(US, 5)
states

CO = conus.loc[US['state'] == 'Colorado']
counties = sample_nsrdb(CO, 5)
counties

Denver = CO.loc[CO['county'] == 'Denver']
pixels = sample_nsrdb(Denver, 5)
pixels

