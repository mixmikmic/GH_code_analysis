import ipyvolume

ds = ipyvolume.datasets.aquariusA2.fetch()
ipyvolume.quickvolshow(ds.data, lighting=True)



