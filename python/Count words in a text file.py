par = """
The Time Traveller (for so it will be convenient to speak of him)
was expounding a recondite matter to us. His grey eyes shone and
twinkled, and his usually pale face was flushed and animated. The
fire burned brightly, and the soft radiance of the incandescent
lights in the lilies of silver caught the bubbles that flashed and
passed in our glasses. Our chairs, being his patents, embraced and
caressed us rather than submitted to be sat upon, and there was that
luxurious after-dinner atmosphere when thought roams gracefully
free of the trammels of precision. And he put it to us in this
way--marking the points with a lean forefinger--as we sat and lazily
admired his earnestness over this new paradox (as we thought it)
and his fecundity.
"""

import string
string.punctuation

translator = str.maketrans({key: None for key in string.punctuation})
par.translate(translator)

words = par.translate(translator).lower().split()

words

frequency = {}  # empty dictionary
for word in words:
    if word not in frequency:
        frequency[word] = 1
    else:
        frequency[word] += 1

frequency

fin = open("pg35.txt")
frequency = {}
# Read file line by line
for line in fin:
    words = line.translate(translator).lower().split()
    for word in words:
        if word not in frequency:
            frequency[word] = 1
        else:
            frequency[word] += 1
fin.close()

frequency.items()
sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:30]



