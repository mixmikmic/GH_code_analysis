from ga4gh.client import client
c = client.HttpClient("http://1kgenomes.ga4gh.org")

dataset = c.search_datasets().next()
print dataset
data_set_id = dataset.id

dataset_via_get = c.get_dataset(dataset_id=data_set_id)
print dataset_via_get

