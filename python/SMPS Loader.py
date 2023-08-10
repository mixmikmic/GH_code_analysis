import atmospy

print ("atmospy v{}".format(atmospy.__version__))

# Setup an instance of the io.SMPS class
raw = atmospy.io.SMPS()

# Load the file of choice
get_ipython().magic('time raw.load("../tests/data/SMPS_Number.txt")')

get_ipython().magic('time d = atmospy.aerosols.ParticleDistribution(histogram = raw.histogram, bins = raw.bins)')

get_ipython().magic('time d.compute()')

get_ipython().magic('time d.statistics()')

d.stats['Number'].head()

d.data['dN'].head()

d.meta.head()



