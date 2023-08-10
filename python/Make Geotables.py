# Click the Blue Plane to preview this notebook as a CrossCompute Tool
source_table_path = 'selected-features.csv'
target_folder = '/tmp'

from os.path import join
from shutil import copy
target_path = join(target_folder, 'examples.csv')
copy(source_table_path, target_path)
print('example_satellite_geotable_path = ' + target_path)

