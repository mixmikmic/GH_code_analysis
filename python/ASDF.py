import numpy as np
from astropy.modeling import models
from astropy import coordinates
from astropy import table
from astropy.io import fits, ascii
from asdf import AsdfFile

f = AsdfFile()
print(f.tree)

f.tree['model'] = models.Rotation2D(angle=23)
f.write_to('rotation.asdf')
get_ipython().system('less rotation.asdf')

fa = AsdfFile.open('rotation.asdf')
model = fa.tree['model']
print(model(1, 2))

f = AsdfFile()
f.tree['model'] = models.Gaussian1D(amplitude=10, mean=3, stddev=.3)

f.write_to("gauss.asdf")

a = np.random.rand(2,3)

f = AsdfFile()
f.tree['data1'] = a
f.write_to('array1.asdf')
get_ipython().system('less array1.asdf')

f = AsdfFile()
f.tree['data'] = a
f.write_to('array.asdf', auto_inline=20)
get_ipython().system('less array.asdf')

a1 = a[:1, 2:]
f = AsdfFile()
f.tree['full_array'] = a
f.tree['array_subset'] = a1
f.write_to("shared_array.asdf")#, auto_inline=20)
get_ipython().system('less shared_array.asdf')
           

f.write_to('external_array.asdf', all_array_storage='external')
get_ipython().system('ls external_array*')

get_ipython().system('less external_array.asdf')

get_ipython().system('less external_array0000.asdf')

fa=AsdfFile.open('external_array.asdf')

print fa.tree['array_subset']#.copy()

fa.add_history_entry("Source CDP-5", {'name': "jreftools", "author": 'STScI',
                                       "homepage": "http://github.com/spacetelescope/jreftools",
                                       "version": "0.6"})
fa.write_to("with_history.asdf")
get_ipython().system('less with_history.asdf')

