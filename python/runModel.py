import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = FutureWarning)

from pyBadlands.model import Model as badlandsModel

# Initialise model
model = badlandsModel()

# Define the XmL input file
model.load_xml('input.xml')

model.run_to_time(130000)





