import os
from fileoperations.fileoperations import get_filenames_in_dir
from tomato.symbolic.scoreconverter import ScoreConverter

mu2folder = os.path.join('..', '..', 'mu2')
xmlfolder = os.path.join('..', '..', 'MusicXML')

mu2filepaths, dummyfolders, mu2files = get_filenames_in_dir(mu2folder, '*.mu2')
symbtrnames = [os.path.splitext(sf)[0] for sf in mu2files]
xmlfilepaths = [os.path.join(xmlfolder, sn + '.xml') for sn in symbtrnames]

for ii, (mf, xf) in enumerate(zip(mu2filepaths, xmlfilepaths)):
    print str(ii) + ' ' + sn
    ScoreConverter.mu2_to_musicxml(mf, xml_out=xf)

