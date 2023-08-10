import pandas as pd

def get_historical_data():
    return pd.concat([

        # 2017-10
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/f7564618-2d92-432e-96a8-c7a29643cf1d/download/Service-Station--Price-History-October-2017.xlsx", skiprows=1),

        # 2017-09
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/8b6a7a49-3cb2-4501-97dd-344ac0883ad6/download/Service-Station--Price-History-September-2017.xlsx", skiprows=1),

        # 2017-08
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/85f33d70-af2e-4c5a-ab03-122df9cbabe4/download/Service-Station--Price-History-August-2017.xlsx", skiprows=1),

        # 2017-07
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/d59adf5e-bcf6-4b0c-82a6-41ac9ec9162a/download/Service-Station--Price-History-July-2017.xlsx", skiprows=1),

        # 2017-06
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/dba9405e-ad7e-4280-b994-041485db0e88/download/Service-Station--Price-History-June-2017.xlsx"),

        # 2017-05
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/a23dc2e7-8ca7-422d-9603-6c5693374318/download/Service-Station--Price-History-May-2017.xlsx"),

        # 2017-04
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/28a5e738-5fae-4e74-84dd-20adf0488d86/download/Service-Station-and-Price-History-April-2017.xlsx"),

        # 2017-03
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/5ad2ad7d-ccb9-4bc3-819b-131852925ede/download/Service-Station-and-Price-History-March-2017.xlsx"),

        # 2017-02
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/f6414eb2-26ac-405b-8d1c-79680074f851/download/Service-Station-and-Price-History-February-2017.xlsx"),

        # 2017-01
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/30d9d13d-ff8e-4041-82a1-1aa909d38f65/download/Service-Station-and-Price-History-January-2017.xlsx"),

        # 2016-12
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/2a7128ae-02fa-40f7-b9de-a75479ebc9e4/download/PriceHistoryDec2016.xlsx", skiprows=1),

        # 2016-11
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/d8e32bc9-9561-4971-abd5-21862f50d60d/download/PriceHistoryNov2016.xlsx", skiprows=1),

        # 2016-10
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/7b09946e-ffa8-45f0-90b9-36b90af6e510/download/Service-Stations-and-Price-History-October-2016.xlsx"),

        # 2016-09
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/6d0a644f-83d8-49b2-beef-4fb180e4f6d1/download/Service-Station-and-Price-History--September-2016.xlsx"),

        # 2016-08
        pd.read_excel("https://data.nsw.gov.au/data/dataset/a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b/resource/efebafce-5ddf-4f85-9840-07654b01a7a2/download/Service-Station-and-Price-History--August-2016.xlsx").rename(columns={'FuelType': 'FuelCode'})
    
    ], ignore_index=True)

# Use the function above to download all of the data, then write it to a CSV

get_historical_data().to_csv('../data/price_history.csv.gz', compression='gzip', index=False)

# Check we can read it!

pd.read_csv('../data/price_history.csv.gz', compression='gzip').head()

