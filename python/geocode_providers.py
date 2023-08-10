import pandas as pd
import numpy as np

from backend.models import handle_points
from backend.lib import geocoder

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

csv_path = '../../data/all_pcps_addresses.csv'
providers = handle_points.fetch_provider_addresses(csv_path)

addresses = list(providers.full_address)

geocodiocoder = geocoder.GeocodioCoder()
addresses.pop(2338) # This address raises a random 403 error

# Geocode the first 2500 due to Geocodio Limits
results = geocodiocoder.geocode_batch(addresses[:2499])

geocoded_df = pd.DataFrame(results, columns=['full_address', 'lat_long'])

geocoded_df.to_csv('1st-2500-geocoded-pointBs-93710.csv', index=False)

