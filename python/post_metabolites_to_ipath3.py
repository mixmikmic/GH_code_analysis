get_ipython().run_line_magic('pylab', 'inline')
import pandas as pd
import ipath
from IPython.display import SVG

ids= ['C05499', 'C20359', 'C07207', 'C00022', 'C19712', 'C03564',
       'C15520', 'C05810', 'C05500', 'C14779', 'C00684', 'C15503',
       'C17339', 'C01401', 'C02939', 'C05455', 'C00736', 'C04734',
       'C10208', 'C05198', 'C02237', 'C12110', 'C00309', 'C00158',
       'C00855', 'C04916', 'C07577', 'C00310', 'C16649', 'C00184',
       'C00140', 'C02139', 'C14180', 'C00049', 'C09712', 'C02714',
       'C16618', 'C00123', 'C04785', 'C04020', 'C19585', 'C07394',
       'C05813', 'C00410', 'C05635', 'C02630', 'C14774', 'C03344',
       'C03943', 'C02763']

ipath.get_map('\n'.join(ids),keep_colors=True)

ipath.scale_map('map')
SVG(data='map_scaled.svg')


D= pd.DataFrame(index=ids)
D['log2FC']=5*np.random.randn(D.shape[0])
D['p_values'] = 10.**(-np.random.randint(0,high=5,size=D.shape[0]))
D.head()

selection= ipath.create_selection(D,color_column='log2FC',width_column='p_values')

print(selection[:300]+'...')

help(ipath.to_parameters)



ipath.get_map(selection)

ipath.scale_map('map')
SVG(data='map_scaled.svg')



