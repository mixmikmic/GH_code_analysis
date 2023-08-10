import glob

fnames = glob.glob("../MusicXML_files/wikifonia20100503/*.xml")
fnames[:10]

import xml.etree.cElementTree as ET

tree = ET.ElementTree(file=fnames[0])

root = tree.getroot()
root

root.getchildren()

root.find('identification/creator').text

root.find('movement-title').text

def title_composer(fname):
    "Returns title and composer name from XML filename."
    root = ET.ElementTree(file=fname).getroot()
    return (root.find('identification/creator').text, root.find('movement-title').text)
    

title_composer(fnames[0])

metadata = [title_composer(fname) for fname in fnames]

import pandas as pd

df = pd.DataFrame(data=metadata, index=fnames, columns=('composer', 'song_title'))
df.head(10)

df.shape

df[df.composer.str.contains('rolling', case=False)]

df[df.composer.str.contains('stone', case=False)]

df[df.composer.str.contains('keith', case=False)]

df[df.composer.str.contains('lennon', case=False)]



