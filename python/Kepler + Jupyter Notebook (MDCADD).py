import KeplerMagicFunction

get_ipython().magic('KpConf /Users/spurawat/Kepler_Repository/bioKepler-2017/kepler.modules/kepler.sh')

get_ipython().magic('WpConf /Users/spurawat/nbcr/rocce-NBCR-Product/MDCADD.xml')

get_ipython().magic('Kepler')

graph = '/Users/spurawat/NBCR_Demo_Feb17/IPython-Kepler-Magic-Function/p53_zinc07135644/prod2_total_energy.png' 
from IPython.display import Image
Image(filename=graph) 

output = open('MDCADD-Report.txt','r')
print(output.read())



