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

note1 = note.Note("G3")     # declaration of first note
note2 = note.Note("B-3")
note3 = note.Note("D4")
note4 = note.Note("F#4")
note5 = note.Note("A4")
note6 = note.Note("C5")
note7 = note.Note("E5")
note8 = note.Note("G#5")
note9 = note.Note("B5")
note10 = note.Note("C#6")
note11 = note.Note("D#6")
note12 = note.Note("F6")

# combine the twelve notes in a row list
bergRow = [note1, note2, note3, note4, note5, note6, note7, note8, note9, note10, note11, note12]
bergRow    # output of bergRow (by just using the name of the variable)

dir(note)

for currentNote in bergRow:                                    # for every note in bergRow list do...
    currentNote.duration.type = 'whole'                        # ... declare duration of a whole note
    print(currentNote.duration, currentNote.nameWithOctave)    # ... output of note duration and name (using the print command)

bergStream = stream.Stream()        # create empty stream

for currentNote in bergRow:         # iterate over every note in bergRow and ...
    bergStream.append(currentNote)  # ... append current note to the stream

bergStream.show('text')             # output of the stream (using the .show()-method with option 'text'; compare to output above)

len(bergStream)

len(bergStream.flat.getElementsByClass(note.Note))

# bergStream.show()
render(bergStream)

bergStream.show('lily.pdf')

bergStream.show('lily.png')

bergRowTiny = converter.parse("tinyNotation: G1 B- d f# a c' e' g'# b' c''# d''# f''")
# bergRowTiny.show()
render(bergRowTiny)

bergRowTiny.show('text')

bergStream.analyze('ambitus')

bergStream.analyze('key')

# declare some variables as Chord()-Objects
triad1 = chord.Chord()
triad2 = chord.Chord()
triad3 = chord.Chord()
triad4 = chord.Chord()
wtScale = chord.Chord()

# iterate over the first three notes in the stream
for currentNote in bergStream[0:3]:
    triad1.add(currentNote)           # add the currentNote to the Chord()

# ...
for currentNote in bergStream[2:5]:
    triad2.add(currentNote)

# ...
for currentNote in bergStream[4:7]:
    triad3.add(currentNote)

# ...
for currentNote in bergStream[6:9]:
    triad4.add(currentNote)

# iterate over the last three notes in the stream
for currentNote in bergStream[8:12]:
    wtScale.add(currentNote)

# output the 5 chords
# triad1.show()
# triad2.show()
# triad3.show()
# triad4.show()
# wtScale.show()

render(triad1)
render(triad2)
render(triad3)
render(triad4)
render(wtScale)

fullChord = chord.Chord([triad1, triad2, triad3, triad4, wtScale])

# fullChord.show()
render(fullChord)

# create empty stream
chordsStream = stream.Stream()

# append all the triads to the stream
chordsStream.append(triad1);
chordsStream.append(triad2);
chordsStream.append(triad3);
chordsStream.append(triad4);
chordsStream.append(wtScale);

# chordsStream.show()
render(chordsStream)

# iterate over every chord in the stream, and ...
for currentChord in chordsStream:
    currentChord.addLyric(currentChord.pitchedCommonName)    # ... add triad name
    currentChord.addLyric(currentChord.intervalVector)       # ... add interval vector
    currentChord.addLyric(currentChord.primeForm)            # ... add prime form
    currentChord.addLyric(currentChord.forteClass)           # ... add forte class

# chordsStream.show()
render(chordsStream)

for currentChord in chordsStream.recurse().getElementsByClass('Chord'):
    if currentChord.forteClass == '3-11A':
        currentChord.style.color = 'red'
        for x in currentChord.derivation.chain():
            x.style.color = 'blue'
    if currentChord.forteClass == '3-11B':
        currentChord.style.color = 'blue'
        for x in currentChord.derivation.chain():
            x.style.color = 'blue'

# chordsStream.show()
render(chordsStream)



sorted(list(serial.historicalDict))

bergRowInternal = serial.getHistoricalRowByName('RowBergViolinConcerto')
print(type(bergRowInternal))
print(bergRowInternal.composer)
print(bergRowInternal.opus)
print(bergRowInternal.title)
print(bergRowInternal.row)
print(bergRowInternal.pitchClasses())
bergRowInternal.noteNames()

g = bergRowInternal.originalCenteredTransformation('T', 0)
u = bergRowInternal.originalCenteredTransformation('I', 0)
k = bergRowInternal.originalCenteredTransformation('R', 0)
ku = bergRowInternal.originalCenteredTransformation('RI', 0)

print('original:')
# g.show()
render(g)

print('inversion:')
# u.show()
render(u)

print('retrograde:')
# k.show()
render(k)

print('retrograde inversion:')
#ku.show()
render(ku)

bergMatrix1 = bergRowInternal.matrix()
print(bergMatrix1)

bergMatrix2 = serial.rowToMatrix(bergRowInternal.row)
print(bergMatrix2)

segmentationList = {}
segmentationLength = 3     # here you can choose the length of the segments (try other values)

rangeEnd = 12 - segmentationLength + 1


# iterate over the whole tone row in (rangeEnd - 1) steps
for i in range(0,rangeEnd):
    print('---')
    # create an empty placeholder for the segment as a ToneRow()-Object 
    # at the position i in the segmentationList
    segmentationList[i] = serial.ToneRow()
    
    # fill up the segment with the corresponding notes
    for currentNote in bergRowInternal[i:i+segmentationLength]:
        segmentationList[i].append(currentNote)
    print('Run ', i, ' completed.')     # This is for control only.
    
segmentationList     # output of the whole list

# check for triads in the segmentation list
# make sure to use segmentLength = 3 above 
# (for segmentLength = 4 you will get 7th and other tetra chords)

for i in segmentationList:
    print('---')
    print('RUN ', i)
    outputString = ''
    
    # get a list of the pitches of the current segment
    currentPitchList = segmentationList[i].pitches
    print(currentPitchList)
    
    #use the pitchList as input for a chord
    currentChord = chord.Chord(currentPitchList)
    
    # check for minor triad (with highlighting)
    # use forteClass 3-11A instead of 'isMinorTriad()'-method to catch enharmonic equivalents
    if currentChord.forteClass == '3-11A':        
        currentChord.style.color = 'red'
        outputString = 'MINOR TRIAD: '
    
    # check for major triad (with highlighting)
    # use forteClass 3-11B instead of 'isMajorTriad()'-method to catch enharmonic equivalents
    if currentChord.forteClass == '3-11B':
        currentChord.style.color = 'blue'
        outputString = 'MAJOR TRIAD: '
    
    # currentChord.show()
    render(currentChord)
    
    outputString += currentChord.pitchedCommonName
    print(outputString)





