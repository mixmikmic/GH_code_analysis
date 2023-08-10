import pandas as pd

md_base_url = "https://data.cms.gov/resource/4uvc-gbfz" # URL For The Data

count_url = md_base_url + "?" + "$select=count(*)"
print(count_url)
pd.read_json(count_url)

count_by_state_url = md_base_url + "?" + "$select=nppes_provider_state,count(nppes_provider_state)&$group=nppes_provider_state"
print(count_by_state_url)
by_state_df = pd.read_json(count_by_state_url)
by_state_df[by_state_df.count_nppes_provider_state < 50000]

import requests

r = requests.get(md_base_url, params={'$where': "nppes_provider_state='DC'", "$limit": 50000})

r.headers

prescriber_df = pd.read_json(r.content)

# Uncomment if you are unable to install the requests library
# prescriber_df = pd.read_json(prescriber_url)

prescriber_df.head(10)

prescriber_df = prescriber_df.sort_values(by=["specialty_desc","npi","drug_name"])

len("1487818670")

r = requests.get("http://www.bloomapi.com/api/search", 
                 params={"limit": 10,"offset": 0, "key1": "npi", "op1": "eq", "value1": 1487818670})
r.json()

# prescriber_df.to_csv("./data/medicare_dc_prescriber_raw_2013.csv")
# Uncomment last line if there are API Connection Problems
# prescriber_df = pd.read_csv("./data/medicare_dc_prescriber_raw_2013.csv")

import numpy as np

npi_drug_cross_df = pd.crosstab(prescriber_df["npi"], prescriber_df["drug_name"], values=prescriber_df["total_claim_count"], 
                                  aggfunc=np.sum)

npi_drug_cross_df.head(5)

npi_drug_cross_df = npi_drug_cross_df.fillna(0)

npi_drug_cross_df["ABILIFY"].sum()

prescriber_df["drug_count"] = 1

prescriber_specialty_df = prescriber_df.groupby(["npi","specialty_desc"]).agg({"total_claim_count": np.sum,  "drug_count": np.sum})

prescriber_specialty_df.reset_index(level=["npi","specialty_desc"], inplace=True)

prescriber_specialty_df.head(10)

prescriber_specialty_drugs_df = pd.merge(prescriber_specialty_df, npi_drug_cross_df.reset_index(level=["npi"]), on="npi")

prescriber_specialty_drugs_df.head(5)

import seaborn as sb

get_ipython().magic('matplotlib inline')

sb.boxplot(x="specialty_desc", y="drug_count", 
           data=prescriber_specialty_drugs_df[(prescriber_specialty_drugs_df["specialty_desc"] == "Cardiology") |
           (prescriber_specialty_drugs_df["specialty_desc"] == "Internal Medicine") |
           (prescriber_specialty_drugs_df["specialty_desc"] == "Psychiatry & Neurology")]
          )

npi_generic_cross_df = pd.crosstab(prescriber_df["npi"], prescriber_df["generic_name"], values=prescriber_df["total_claim_count"], 
                                  aggfunc=np.sum)

npi_generic_cross_df = npi_generic_cross_df.fillna(0)

npi_generic_cross_df.head(5)

npi_generic_cross_df.columns

prescriber_specialty_generic_df = pd.merge(prescriber_specialty_df, npi_generic_cross_df.reset_index(level=["npi"]), on="npi")

prescriber_specialty_generic_df.head(5)

prescriber_specialty_generic_df = prescriber_specialty_generic_df.sort_values(by=["specialty_desc", "npi"])

pf_matrix = prescriber_specialty_generic_df.iloc[: , 4: ].as_matrix()

pf_matrix.shape

identifiers_matrix = prescriber_specialty_df.iloc[:,0:2].as_matrix()

generic_array = prescriber_specialty_generic_df.columns[4:]

import scipy.spatial

normalized_pf_matrix = pf_matrix / np.reshape(np.sum(pf_matrix,1), (pf_matrix.shape[0],1))

normalized_pf_matrix.shape

prescriber_dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(normalized_pf_matrix, "euclidean"))

prescriber_dist.shape

import matplotlib.pyplot as plt

plt.matshow(prescriber_dist)

prescriber_dist[2010,:]

plt.scatter(np.arange(prescriber_dist.shape[0]),prescriber_dist[2010,:])

plt.scatter(np.arange(prescriber_dist.shape[0]),np.sort(prescriber_dist[2010,:]))

prescriber_specialty_generic_df.iloc[2010,0:4]

providers_sorted = np.lexsort((prescriber_dist[:,2010].tolist(),))

prescriber_specialty_generic_df.iloc[:,0:2].as_matrix()[providers_sorted[0:40],:]

np.lexsort(((-1 * pf_matrix[2010,:]).tolist(),))[0:17]

generic_array[np.lexsort(((-1 * pf_matrix[2010,:]).tolist(),))][0:16]

prescriber_df[prescriber_df["npi"]==1487818670].sort_values("total_claim_count", ascending=False)

#prescriber_specialty_drugs_df.to_csv("./data/medicare_dc_providers_generic_prescribing_2013.csv")

import h5py

f5 = h5py.File("./data/medicare_dc_prescriber_matrix_2013.hdf5", "w")

ds_raw_count = f5.create_dataset("/generic/raw_counts/core_array/", shape=pf_matrix.shape, dtype=pf_matrix.dtype)
ds_pdist = f5.create_dataset("/prescriber/generic/distance/", shape=prescriber_dist.shape, dtype=prescriber_dist.dtype)
ds_normalized = f5.create_dataset("/generic/normalized/core_array/", shape=normalized_pf_matrix.shape, dtype=normalized_pf_matrix.dtype)
ds_identifiers = f5.create_dataset("/prescriber/identifiers/", shape=identifiers_matrix.shape, dtype="S128")
ds_generic_name = f5.create_dataset("/generic/names", shape=generic_array.shape, dtype="S128")

ds_raw_count[...] = pf_matrix
ds_normalized[...] = normalized_pf_matrix
ds_pdist[...] = prescriber_dist

ds_identifiers[...] = np.array(identifiers_matrix.tolist(), dtype="S128")

ds_generic_name[...] = np.array(generic_array.tolist(), dtype="S128")

f5.close()

np.array(generic_array.tolist(), dtype="S128")[0:20]

pd.read_json("https://data.medicare.gov/resource/3uxj-hea6?NPI=1487818670")

npi_with_distance = pd.DataFrame({"dist": prescriber_dist[2010,:], "npi": prescriber_specialty_generic_df["npi"],  
                                  "specialty_desc": prescriber_specialty_generic_df["specialty_desc"]})

details_df = pd.read_json("https://data.medicare.gov/resource/3uxj-hea6?NPI=1487818670")
for npi in npi_with_distance.sort_values("dist")["npi"][1:50]:
    details_df = details_df.append(pd.read_json("https://data.medicare.gov/resource/3uxj-hea6?NPI=" + str(npi)))

details_df.head(5)

pd.merge(details_df, npi_with_distance, on="npi")

#prescriber_specialty_drugs_df.groupby("specialty_desc").agg({"specialty_desc": np.size}).sort_values("specialty_desc", ascending=False)

