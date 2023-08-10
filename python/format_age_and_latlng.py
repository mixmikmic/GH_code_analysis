import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from geopy import Nominatim
import folium
from branca.colormap import LinearColormap, StepColormap

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('../data/round4/secondary_run1.csv')

df.head()

df.describe()

geocoder = Nominatim()
def address_to_latlng(address: str, num_retries: int = 3) -> tuple:
    """
    Convert an address to a point in LatLng with geocoding
    """
    print("search for {}".format(address))
    for i in range(num_retries):
        try:
            location = geocoder.geocode(query=address + ', Vancouver BC')
        except Exception as e:
            time.sleep(2)
            print("Error: {}. Retrying...({} retries left)".format(e, num_retries - i - 1))
            continue
        break
        
    if location is not None:
        print("found {}".format(location))
        return (location.latitude, location.longitude)
latlngs = []
for address in df['address']:
    latlng = address_to_latlng(address=address)
    latlngs.append(latlng)

df['latlng'] = latlngs
df.head()

age_list = []
for age in df['age']:
    try:
        split = age.split('(')
        age = int(split[1].split(' ')[0])
    except Exception as e:
        age = None
    finally:
        age_list.append(age)
df['age'] = age_list

df.to_csv('../data/round4/secondary_run1_clean.csv', index=False)

df2 = pd.DataFrame.from_csv('../data/round3/rew_round3_with_latlng.csv')

# wierd...df2 has address as the index
df2['address'] = df2.index
df2.index = range(len(df2))

df_c = df.append(df2)

df_c = df.append(df2, ignore_index=True)

df_c = df_c.drop_duplicates(subset='address').reset_index()

df_c.to_csv('../data/all_jan12.csv', index=False)

