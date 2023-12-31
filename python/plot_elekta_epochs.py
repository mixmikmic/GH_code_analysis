get_ipython().run_line_magic('matplotlib', 'inline')

# Author: Jussi Nurminen (jnu@iki.fi)
#
# License: BSD (3-clause)


import mne
import os
from mne.datasets import multimodal
from mne import AcqParserFIF

fname_raw = os.path.join(multimodal.data_path(), 'multimodal_raw.fif')


print(__doc__)

raw = mne.io.read_raw_fif(fname_raw)
ap = AcqParserFIF(raw.info)

print(ap)

cond = ap.get_condition(raw, 'Auditory right')
epochs = mne.Epochs(raw, **cond)
epochs.average().plot_topo()

evokeds = []
for cat in ap.categories:
    cond = ap.get_condition(raw, cat)
    # copy (supported) rejection parameters from DACQ settings
    epochs = mne.Epochs(raw, reject=ap.reject, flat=ap.flat, **cond)
    evoked = epochs.average()
    evoked.comment = cat['comment']
    evokeds.append(evoked)
# save all averages to an evoked fiff file
# fname_out = 'multimodal-ave.fif'
# mne.write_evokeds(fname_out, evokeds)

newcat = dict()
newcat['comment'] = 'Visual lower left, longer epochs'
newcat['event'] = 3  # reference event
newcat['start'] = -.2  # epoch start rel. to ref. event (in seconds)
newcat['end'] = .7  # epoch end
newcat['reqevent'] = 0  # additional required event; 0 if none
newcat['reqwithin'] = .5  # ...required within .5 sec (before or after)
newcat['reqwhen'] = 2  # ...required before (1) or after (2) ref. event
newcat['index'] = 9  # can be set freely

cond = ap.get_condition(raw, newcat)
epochs = mne.Epochs(raw, reject=ap.reject, flat=ap.flat, **cond)
epochs.average().plot()

