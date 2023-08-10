from skdiscovery.visualization import spiral_plot

from skdaccess.framework.param_class import *
from skdaccess.astro.kepler import DataFetcher as Kepler_DF
from skdaccess.utilities.kepler_util import normalize

import warnings
warnings.filterwarnings(action='ignore')

kepler_fetcher = Kepler_DF([AutoList(['009941662'])])
kepler_data    = kepler_fetcher.output().get()

normalize(kepler_data['009941662'])

kepler_flux = kepler_data['009941662'].iloc[:10000]
kepler_flux.set_index('TIME',inplace=True)

spiral_plot.run_spiral(kepler_flux['PDCSAP_FLUX'], period = 1.7636,plotTS=True)

spiral_plot.run_spiralInteractive(kepler_flux['PDCSAP_FLUX'],1.7,[1.75,1.77,0.001],plotTS=True)

