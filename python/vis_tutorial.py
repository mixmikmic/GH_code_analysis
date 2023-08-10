get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = (8.0, 8.0)
import plantcv as pcv

class options:
    def __init__(self):
        self.image = "img/tutorial_images/vis/original_image.jpg"
        self.debug = "plot"
        self.writeimg = False
        self.result = "vis_tutorial_results.txt"

# Get options
args = options()

# Read image
img, path, filename = pcv.readimage(args.image, args.debug)

# Pipeline step
device = 0



