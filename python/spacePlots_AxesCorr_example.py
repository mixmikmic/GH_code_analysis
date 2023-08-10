import spaceplot_DMtools as spdmt # tools to load/manipulate distance matrices
import spaceplot_plotter as spp # plotting tools

labels,DM = spdmt.read_DM( 'example_DM.csv' )

help(spp.axesR_plot)
r = -.4
spp.axesR_plot(r, label_axes=['Trustworthiness','Dominance'], orthComp=True )

