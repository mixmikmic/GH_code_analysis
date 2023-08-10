import msaf
import librosa
import seaborn as sns

# and IPython.display for audio output
import IPython.display

# Setup nice plots
sns.set(style="dark")
get_ipython().magic('matplotlib inline')

# Choose an audio file and listen to it
audio_file = "../datasets/Sargon/audio/01-Sargon-Mindless.mp3"
IPython.display.Audio(filename=audio_file)

# Segment the file using the Foote method, and Pitch Class Profiles for the features
results = msaf.process(audio_file, feature="hpcp", boundaries_id="gt", labels_id="fmc2d", plot=True)

# Evaluate the results. It returns a pandas data frame.
evaluations = msaf.eval.process(audio_file, boundaries_id="foote", labels_id="fmc2d", feature="hpcp")
print evaluations

# Listen to the boundaries
out_file = "sonified_bounds.wav"
results = msaf.process(audio_file, feature="hpcp", boundaries_id="foote", sonify_bounds=True, out_bounds=out_file)
# IPython.display.Audio(filename=out_file)

# Label the file using the 2D-FMC method, and Pitch Class Profiles for the features
results = msaf.process(audio_file, feature="hpcp", labels_id="fmc2d", plot=True)

# Evaluate the results. It returns a pandas data frame.
evaluations = msaf.eval.process(audio_file, boundaries_id="gt", labels_id="fmc2d", feature="hpcp")
print evaluations

# First, check which are foote's algorithm parameters:
print msaf.algorithms.foote.config

# play around with IPython.Widgets
from IPython.html.widgets import interact

# Obtain the default configuration
bid = "foote"  # Boundaries ID
lid = None     # Labels ID
feature = "hpcp"
config = msaf.io.get_configuration(feature, annot_beats=False, framesync=False, boundaries_id=bid, labels_id=lid)

# Sweep M_gaussian parameters
@interact(M_gaussian=(50, 500, 25))
def _run_msaf(M_gaussian):
    # Set the configuration
    config["M_gaussian"] = M_gaussian
    
    # Segment the file using the Foote method, and Pitch Class Profiles for the features
    results = msaf.process(audio_file, feature=feature, boundaries_id=bid, config=config, plot=True)

    # Evaluate the results. It returns a pandas data frame.
    evaluations = msaf.eval.process(audio_file, boundaries_id="foote", feature="hpcp", config=config)
    print evaluations



