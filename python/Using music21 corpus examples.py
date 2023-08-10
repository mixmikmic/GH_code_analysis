get_ipython().run_line_magic('matplotlib', 'inline')
# imports the matplot library to plot graphs etc.

from music21 import *

env = environment.Environment()
# env['musicxmlPath'] = 'path/to/your/musicXmlApplication'
# env['lilypondPath'] = 'path/to/your/lilyPond'
# env['musescoreDirectPNGPath'] = 'path/to/your/museScore'

print('Environment settings:')
print('lilypond: ', env['lilypondPath'])
print('musicXML: ', env['musicxmlPath'])
print('musescore: ', env['musescoreDirectPNGPath'])

demoPaths = corpus.getComposer('demos')
demoPaths

demoPath = demoPaths[0]

demo = corpus.parse(demoPath)

print(demo.corpusFilepath)
demo.show()





sbBundle = corpus.search('Bach', 'composer')
print(sbBundle)
print(sbBundle[0])
print(sbBundle[0].sourcePath)
sbBundle[0].metadata.all()





s = corpus.parse('bach/bwv65.2.xml')
s.show()



fVoices = stream.Part((s.parts['Soprano'], s.parts['Alto'])).chordify()
mVoices = stream.Part((s.parts['Tenor'], s.parts['Bass'])).chordify()

chorale2p = stream.Score((fVoices, mVoices))
chorale2p.show()

upperVoices = stream.Part((s.parts['Soprano'], s.parts['Alto'], s.parts['Tenor'])).chordify()
bass = stream.Part((s.parts['Bass']))

chorale3p = stream.Score((upperVoices, bass))
chorale3p.show()

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

choraleChords.show()









p = corpus.parse('bach/bwv846.xml')
p.show()

p.analyze('key')

p.show('text')

len(p.parts)

len(p.flat.notes)

graph.findPlot.FORMATS

p.plot('pianoroll')

p.plot('horizontalbar')









