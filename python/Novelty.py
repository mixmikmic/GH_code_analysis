from run_novelty import Novelty
n = Novelty()
n.get_novelty("/path/to/text")

from run_novelty import Novelty
n = Novelty()
n.get_novelty("/path/to/text")
n.mark_up_text()

import os
from novelty_filter import NoveltyFilter
directory = "/path/"

for f in os.listdir(directory):

    # Select all .txt files in a given directory, excluding the marked-up files
    # created e.g. in the previous cell.
    if f.endswith(".txt") and "_intervals" not in f:

        full_path = os.path.join(directory, f)
        print(full_path)
        
        # Use delete keyword "False" so that the filter won't be reinitialized for each text.
        n = NoveltyFilter(delete=False)
        
        # The paths to store outputs *must* exist prior to running this code.
        n.save_outputs(full_path, filter_output="outputs/filters/texts/", graph_output="outputs/results/texts/")

