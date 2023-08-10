get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np

annexB = pd.read_excel("../data/raw/rinse-annex-b/AnnexB RINSE Registry of NNS.xlsx", sep="\t", sheetname="Registry")
neobiota = pd.read_excel("../data/raw/rinse/neobiota-023-065-s001.xlsx", sep="\t")

annex_spec = annexB[["Genus", "Species", "Subspecies"]]
neobiota_spec = neobiota[["Genus", "Species"]]

annex_spec = annex_spec.fillna('')
neobiota = neobiota.fillna('')

annex_spec["GenusSpecies"] = annex_spec["Genus"].str.strip().str.lower() +     annex_spec["Species"].str.strip().str.lower() + annex_spec["Subspecies"].str.strip().str.lower()

neobiota_spec["Species"] = neobiota_spec["Species"].str.replace(" ", "")

neobiota_spec["GenusSpecies"] = neobiota_spec["Genus"].str.strip().str.lower() +                                     neobiota_spec["Species"].str.strip().str.lower()

annex_spec["isin"] = annex_spec["GenusSpecies"].isin(neobiota_spec["GenusSpecies"])

notins = annex_spec[annex_spec["isin"] == False]

len(notins)

notins.drop("isin", axis=1).to_csv("annex_not_in_neobiota.txt")

"avipoxvirus" in neobiota_spec["GenusSpecies"].values

annexB= annexB.fillna('')
neobiota = neobiota.fillna('')

annexB["GenusSpecies"] = annexB["Genus"].str.strip().str.lower() +     annexB["Species"].str.strip().str.lower() + annexB["Subspecies"].str.strip().str.lower()

neobiota["Species"] = neobiota["Species"].str.replace(" ", "")

neobiota["GenusSpecies"] = neobiota["Genus"].str.strip().str.lower() +                                 neobiota["Species"].str.strip().str.lower()

annexB["isin"] = annexB["GenusSpecies"].isin(neobiota["GenusSpecies"])

notin_allcol = annexB[annex_spec["isin"] == False]

len(notin_allcol)

notin_allcol.drop(["isin","GenusSpecies"], axis=1).to_csv("annex_not_in_neobiota_allcol.txt", sep='\t', index=False)



