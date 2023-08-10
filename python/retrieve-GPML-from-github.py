import urllib
import pygplates
import os

def pygplates_retrieve_feature_collection(url):
    
    filename,ext = os.path.splitext(url)
    temporary_file = 'tmp%s' % ext 
    
    urllib.urlretrieve(url,temporary_file)
    
    feature_collection = pygplates.FeatureCollection(temporary_file)
    
    return feature_collection

#
# NOTE - for this to work with files from github, you need to 
# 1. use a 'raw.github.com' address instead of just 'github.com'
# 2. remove the 'blob' from the address, if present
# an alternative address would contain ?raw=true at the end, making it harder to strip the file extension off
# (which is a requirement to load the feature collection)

url = 'https://raw.github.com/chhei/Heine_AJES_15_GlobalPaleoshorelines/master/Global_Paleoshorelines_Golonka.gpml'
feature_collection1 = pygplates_retrieve_feature_collection(url)

for feature in feature_collection1:
    print feature.get_name()
    

url = 'https://raw.github.com/GPlates/pygplates-tutorials/master/Data/LIPs_2014.gpmlz'
feature_collection2 = pygplates_retrieve_feature_collection(url)

for feature in feature_collection2:
    print 'Polygon for %s, eruption age around %d Ma' % (feature.get_name(),feature.get_valid_time()[0])
    



