from campaigns import campaigns
dbs = [c for c in campaigns if 'simz' in c]
print dbs

biom_file = '../../loaders/MolecularEcology/BIOM/otu_table_newsiernounclass_wmetadata.biom'
from biom import load_table
table = load_table(biom_file)
print table.ids(axis='sample')
print table.ids(axis='observation')[:5]

nettows = {}
for db in dbs:
    for s in Sample.objects.using(db).filter(sampletype__name='VerticalNetTow'
                ).order_by('instantpoint__activity__name'):
        print s.instantpoint.activity.name, db
        nettows[s.instantpoint.activity.name] = db

stoqs_sample_data = {}
for s, b in [('simz2013c{:02d}_NetTow1'.format(int(n[4:])), n) for n in table.ids()]:
    sps = SampledParameter.objects.using(nettows[s]
                                        ).filter(sample__instantpoint__activity__name=s)
    # Values of BIOM metadata must be strings, even if they are numbers
    stoqs_sample_data[b] = {sp.parameter.name: str(float(sp.datavalue)) for sp in sps}

new_table = table.copy()
new_table.add_metadata(stoqs_sample_data)
with open(biom_file.replace('.biom', '_stoqs.biom'), 'w') as f:
    new_table.to_json('explore_BIOM_data_for_SIMZ.ipynb', f)

import pprint
pp = pprint.PrettyPrinter(indent=4)
print 'Original: ' + biom_file
print '-' * len('Original: ' + biom_file)
pp.pprint(table.metadata()[:2])
print
print 'New: ' + biom_file.replace('.biom', '_stoqs.biom')
print '-' * len('New: ' + biom_file.replace('.biom', '_stoqs.biom'))
pp.pprint(new_table.metadata()[:2])

from IPython.display import Image
Image('../../../doc/Screenshots/Screen_Shot_2015-10-24_at_10.37.52_PM.png')



