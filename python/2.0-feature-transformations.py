get_ipython().magic('matplotlib inline')
# standard
import sys
import os

# pandas
import pandas as pd

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# needed for project imports
sys.path.append(os.path.join(os.getcwd(), "../.."))

# project imports
from housepredictor.extractor import extract_preliminary


# this styling is purely my preference
# less chartjunk
sns.set_context('notebook', font_scale=1.5, rc={'line.linewidth': 2.5})
sns.set(style='ticks', palette='Set2')

KEY_NAMES = [
'AantalBadkamers',
'AantalKamers',
'AantalWoonlagen',
'Aanvaarding',
'Adres',
'AfgekochtDatum',
'BalkonDakterras',
'BijdrageVVE',
'Bijzonderheden',
'Bouwjaar',
'Bouwvorm',
'BronCode',
'EigendomsSituatie',
'Energielabel.Definitief',
'Energielabel.Index',
'Energielabel.Label',
'Energielabel.NietBeschikbaar',
'Energielabel.NietVerplicht',
'ErfpachtBedrag',
'Garage',
'GarageIsolatie',
'GarageVoorzieningen',
'GelegenOp',
'GewijzigdDatum',
'GlobalId',
'HeeftVeiling',
'HoofdTuinType',
'Inhoud',
'Isolatie',
'Koopprijs',
'Ligging',
'ObjectType',
'ObjectTypeMetVoorvoegsel',
'PerceelOppervlakte',
'Perceeloppervlakte',
'PermanenteBewoning',
'Postcode',
'PromoLabel.HasPromotionLabel',
'PromoLabel.PromotionType',
'PromoLabel.RibbonColor',
'PublicatieDatum',
'SchuurBerging',
'SchuurBergingIsolatie',
'SchuurBergingVoorzieningen',
'ServiceKosten',
'Soort-aanbod',
'SoortDak',
'SoortParkeergelegenheid',
'SoortPlaatsing',
'SoortWoning',
'ToonBezichtigingMaken',
'ToonBrochureAanvraag',
'ToonMakelaarWoningaanbod',
'ToonReageren',
'TuinLigging',
'Verwarming',
'VolledigeOmschrijving',
'Voorzieningen',
'WGS84_X',
'WGS84_Y',
'WarmWater',
'WoonOppervlakte',
'Woonoppervlakte'
]


raw_data = pd.read_json('../data/raw/scrape-results.json')
data = extract_preliminary(raw_data['data'], KEY_NAMES)


def describe_data(data, unique_values=True):
    print("NULL VALUES: ", data.isnull().sum())
    print("HEAD")
    print("="*40)
    print(data.head())
    if unique_values:
        print("\nUNIQUE_VALUES")
        print("="*40)
        print(data.value_counts())

for col_name in data.columns:
    print("FEATURE =============== {}".format(col_name))
    print("=" * 80)
    describe_data(data[col_name])
    print("=" * 80)
    print("\n\n")

FINAL_KEYS = [
    'AantalBadkamers',
'AantalKamers',
'AantalWoonlagen',
'Aanvaarding',
'Adres',
'AfgekochtDatum',
'BalkonDakterras',
'BijdrageVVE',
'Bijzonderheden',
'Bouwjaar',
'Bouwvorm',
'BronCode',
'EigendomsSituatie',
'Energielabel.Definitief',
'Energielabel.Index',
'Energielabel.Label',
'Energielabel.NietBeschikbaar',
'Energielabel.NietVerplicht',
'ErfpachtBedrag',
'Garage',
'GarageIsolatie',
'GarageVoorzieningen',
'GelegenOp',
'GlobalId',
'Inhoud',
'Isolatie',
'Koopprijs',
'Ligging',
'PerceelOppervlakte',
'Perceeloppervlakte',
'PermanenteBewoning',
'Postcode',
'PublicatieDatum',
'SchuurBerging',
'SchuurBergingIsolatie',
'SchuurBergingVoorzieningen',
'ServiceKosten',
'Soort-aanbod',
'SoortDak',
'SoortParkeergelegenheid',
'SoortPlaatsing',
'SoortWoning',
'TuinLigging',
'Verwarming',
'VolledigeOmschrijving',
'Voorzieningen',
'WGS84_X',
'WGS84_Y',
'WarmWater',
'WoonOppervlakte',
'Woonoppervlakte'
]

