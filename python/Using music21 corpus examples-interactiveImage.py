get_ipython().run_line_magic('matplotlib', 'inline')
# imports the matplot library to plot graphs etc.

from music21 import *

# definition of environment settings is different from the settings 
# when this jupyter notebook runs locally on your machine.
# Changes are necessary because jupyter notebook is running via Binder image

env = environment.Environment()

env['lilypondPath']='/usr/bin/lilypond'
env['musescoreDirectPNGPath'] = '/usr/bin/musescore'
env['musicxmlPath'] = '/usr/bin/musescore'

environment.set('pdfPath', '/usr/bin/musescore')
environment.set('graphicsPath', '/usr/bin/musescore')

print('Environment settings:')
print('musicXML:  ', env['musicxmlPath'])
print('musescore: ', env['musescoreDirectPNGPath'])
print('lilypond:  ', env['lilypondPath'])

# re-definition of sho()-method ---> "HACK" from https://github.com/psychemedia/showntell/blob/music/index_music.ipynb
# see also this music21 issue: https://github.com/cuthbertLab/music21/issues/260
get_ipython().run_line_magic('load_ext', 'music21.ipython21')

from IPython.display import Image

def render(s):
    s.show('lily.png')
    return Image(filename=s.write('lily.png'))

demoPaths = corpus.getComposer('demos')
demoPaths

demoPath = demoPaths[0]

demo = corpus.parse(demoPath)

print(demo.corpusFilepath)
#demo.show()
render(demo)





sbBundle = corpus.search('Bach', 'composer')
print(sbBundle)
print(sbBundle[0])
print(sbBundle[0].sourcePath)
sbBundle[0].metadata.all()





s = corpus.parse('bach/bwv65.2.xml')
# s.show()
render(s) 



fVoices = stream.Part((s.parts['Soprano'], s.parts['Alto'])).chordify()
mVoices = stream.Part((s.parts['Tenor'], s.parts['Bass'])).chordify()

chorale2p = stream.Score((fVoices, mVoices))
# chorale2p.show()
render(chorale2p)

upperVoices = stream.Part((s.parts['Soprano'], s.parts['Alto'], s.parts['Tenor'])).chordify()
bass = stream.Part((s.parts['Bass']))

chorale3p = stream.Score((upperVoices, bass))
# chorale3p.show()
render(chorale3p)

chorale3p.show('text')

for c in chorale3p.recurse().getElementsByClass('Chord'):
    print(c)







# chordify the chorale
choraleChords = chorale3p.chordify()

for c in choraleChords.recurse().getElementsByClass('Chord'):
    # force closed position
    c.closedPosition(forceOctave=4, inPlace=True)
    
    # apply roman numerals
    rn = roman.romanNumeralFromChord(c, key.Key('A'))
    c.addLyric(str(rn.figure))
    
    # highlight dimished seventh chords
    if c.isDiminishedSeventh():
        c.style.color = 'red'
    
    # highlight dominant seventh chords
    if c.isDominantSeventh():
        c.style.color = 'blue'

# choraleChords.show()
render(choraleChords)









p = corpus.parse('bach/bwv846.xml')
# p.show()
render(p)

p.analyze('key')

p.show('text')

len(p.parts)

len(p.flat.notes)

graph.findPlot.FORMATS

p.plot('pianoroll')

p.plot('horizontalbar')









