# allow import without install
import sys
if ".." not in sys.path:
    sys.path.append("..")

from jp_gene_viz import fs_chart

w = fs_chart.explore_directory("../..")

w.results



