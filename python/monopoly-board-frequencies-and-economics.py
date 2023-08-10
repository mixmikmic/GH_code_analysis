
get_ipython().system('pip install git+git://github.com/datadotworld/data.world-py.git#egg=project[pandas]')


api_access_token = 'eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJwcm9kLXVzZXItY2xpZW50OmRiYWJiaXR0IiwiaXNzIjoiYWdlbnQ6ZGJhYmJpdHQ6OjU1YjY5MDg2LTQwMGQtNGM1MC1hOGIzLTM2YTVkOTcwMmRjMiIsImlhdCI6MTUxNzM2OTU1Mywicm9sZSI6WyJ1c2VyX2FwaV9yZWFkIiwidXNlcl9hcGlfd3JpdGUiXSwiZ2VuZXJhbC1wdXJwb3NlIjp0cnVlfQ.uD1MUgs0nvvj4-VyWyM_YFTiJ-ag4R-hEda3mA9c_lea6zdJWO-z53Da4ciMIXImJIjEEI6nbkZGJOWjhvEAVg'


get_ipython().system('dw configure')


import datadotworld as dw


# Datasets are referenced by their path
dataset_key = 'dataremixed/monopoly-board-frequencies-and-economics'

# Or simply by their URL
dataset_key = 'https://data.world/dataremixed/monopoly-board-frequencies-and-economics'


# Load dataset (onto the local file system)
dataset_local = dw.load_dataset(dataset_key)  # cached under ~/.dw/cache

# See what is in it
dataset_local.describe()

