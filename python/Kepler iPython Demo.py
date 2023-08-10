import KeplerMagicFunction

get_ipython().magic('KpConf /Applications/Kepler-2.5/Kepler.app/Contents/Resources/Java/kepler.sh')

get_ipython().magic('WpConf /Users/spurawat/kepler-ipython/IPython-Kepler-Magic-Function/chisquare-plot.xml')

get_ipython().magic('Kepler -iterations 200')

get_ipython().magic('readoutput chisquare-plot.SequencePlotter.png')



