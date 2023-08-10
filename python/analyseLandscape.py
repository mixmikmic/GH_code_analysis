import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = FutureWarning)

from scripts import analyseTopo as analyse

# display plots in SVG format
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('matplotlib', 'inline')

foldername = 'TCmodel'

time,island,plateau = analyse.readDataset(folder=foldername, isldPos=[75000.,25000.],isldRadius=2250., 
                                     refPos=[30000.,25000.], pltRadius=10000)

analyse.elevationChange('model T&C',time,island,plateau,figsave=None)

analyse.cumulativeErosion('model T&C',time,island,plateau,figsave=None)

analyse.erosionRate('model T&C',time,island,plateau,figsave=None)



