import numpy as np
import pandas as pd
#-Import custom import methods for the csv files-#
from Functions import (  import_csv_BondList,
                        import_csv_BondTimeSeries)

BondListC = pd.read_csv('DataFrames/BondList.csv', index_col=0)
BondListC.head()

BondQuantC = pd.read_csv('DataFrames/BondQuant.csv')
BondQuantC.head()

BondListC = import_csv_BondList('DataFrames/BondList.csv')
BondListC['First Issue Date'].head()

BondQuantC = import_csv_BondTimeSeries('DataFrames/BondQuant.csv')
BondQuantC.head()

BondPriceC = import_csv_BondTimeSeries('DataFrames/BondPrice.csv')
BondPriceC.head()

## Open the hdf5 file
Bondh5 = pd.HDFStore("DataFrames/BondDF.h5",mode="r")

## Extract and print information on the dataframe
infoh5 = Bondh5.info()
print(infoh5)

BondListH = Bondh5["BondList"]
BondQuantH = Bondh5["BondQuant"]
BondPriceH = Bondh5["BondPrice"]

## Close the hdf5 file
Bondh5.close()

BondListH.head()

BondQuantH.head()

BondPriceH.head()

