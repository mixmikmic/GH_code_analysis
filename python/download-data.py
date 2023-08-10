import urllib as url
import os.path as op

top_url = 'https://stacks.stanford.edu/file/druid:ng782rw8378/'
for i in range(2):
    for ext in ['.nii.gz', '.bvecs', '.bvals']:
        fname = 'SUB1_b2000_%s%s'%(i+1, ext)
        if not op.exists('./data/' + fname):
            url.urlretrieve(op.join(top_url, fname), './data/' + fname)

fnames = ['SUB1_LV1.nii.gz', 'SUB1_aparc-reduced.nii.gz', 'SUB1_t1_resamp.nii.gz']
for fname in fnames:
    if not op.exists('./data/' + fname):
        url.urlretrieve(op.join(top_url, fname), './data/' + fname)



