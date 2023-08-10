import string
from functools import partial

N_ASCII_CHARACTERS = 1 << 7

string.printable

s = {chr(c) for c in range(0, N_ASCII_CHARACTERS) if chr(c).isprintable()}
len(s), ''.join(sorted(s))

s = ''.join(c for c in string.printable if ord(' ') <= ord(c) <= ord('~'))
len(s), s

good_characters = {chr(c) for c in range(ord(' '), ord('~')+1)} | {'\t'}

len(good_characters), good_characters

filename = '20150223-cohpy-memoization.ipynb'

def pass_good_characters(lines):
    for line in lines:
        yield ''.join(
            c for c in line
            if 31 < ord(c) < 127 or c == '\t')

get_ipython().magic('timeit list(pass_good_characters(open(filename)))')

def pass_good_characters(lines):
    for line in lines:
        yield ''.join(
            c for c in line
            if ord(' ')-1 < ord(c) < ord('~')+1 or c == '\t')

get_ipython().magic('timeit list(pass_good_characters(open(filename)))')

def pass_good_characters(lines):
    for line in lines:
        yield ''.join(
            c for c in line
            if ord(' ') <= ord(c) <= ord('~') or c == '\t')

get_ipython().magic('timeit list(pass_good_characters(open(filename)))')

def pass_good_characters(lines):
    for line in lines:
        yield ''.join(
            c for c in line
            if ' ' <= c <= '~' or c == '\t')

get_ipython().magic('timeit list(pass_good_characters(open(filename)))')

def pass_good_characters(lines):
    for line in lines:
        yield ''.join(
            c for c in line
            if c <= '~' and (c.isprintable() or c == '\t'))

get_ipython().magic('timeit list(pass_good_characters(open(filename)))')

def pass_good_characters(lines):
    for line in lines:
        yield ''.join(filter(lambda c: c <= '~' and (c.isprintable() or c == '\t'), line))

get_ipython().magic('timeit list(pass_good_characters(open(filename)))')

def pass_good_characters(lines):
    good_characters = [chr(c) for c in range(ord(' '), ord('~')+1)] + ['\t']
    for line in lines:
        yield ''.join(c for c in line if c in good_characters)

get_ipython().magic('timeit list(pass_good_characters(open(filename)))')

def pass_good_characters(lines):
    good_characters = ''.join([chr(c) for c in range(ord(' '), ord('~')+1)] + ['\t'])
    for line in lines:
        yield ''.join(c for c in line if c in good_characters)

get_ipython().magic('timeit list(pass_good_characters(open(filename)))')

def pass_good_characters(lines):
    good_characters = {chr(c) for c in range(ord(' '), ord('~')+1)} | {'\t'}
    for line in lines:
        yield ''.join(c for c in line if c in good_characters)

get_ipython().magic('timeit list(pass_good_characters(open(filename)))')

def pass_good_characters(lines):
    good_characters = {chr(c) for c in range(ord(' '), ord('~')+1)} | {'\t'}
    yield from (
        ''.join(c for c in line if c in good_characters)
        for line in lines)

get_ipython().magic('timeit list(pass_good_characters(open(filename)))')

def pass_good_characters(lines):
    good_characters = {
        chr(c) for c in range(N_ASCII_CHARACTERS)
        if chr(c).isprintable()} | {'\t'}
    for line in lines:
        yield ''.join(c for c in line if c in good_characters)

get_ipython().magic('timeit list(pass_good_characters(open(filename)))')

def pass_good_characters(lines):
    good_characters = {
        chr(c) for c in range(N_ASCII_CHARACTERS)
        if chr(c).isprintable() or chr(c) == '\t'}
    for line in lines:
        yield ''.join(c for c in line if c in good_characters)

get_ipython().magic('timeit list(pass_good_characters(open(filename)))')

def pass_good_characters(lines):
    good_characters = {chr(c) for c in range(ord(' '), ord('~')+1)} | {'\t'}
    for line in lines:
        yield ''.join(filter(lambda c: c in good_characters, line))

get_ipython().magic('timeit list(pass_good_characters(open(filename)))')

class MyStringIO():
    def __init__(self, s=''):
        self.s = s
        self.i = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        s = []
        for c in iter(partial(self.read, 1), ''):
            s.append(c)
            if c == '\n':
                break
        if not s:
            raise StopIteration
        return ''.join(s)

    def read(self, n):
        s = self.s[self.i:self.i+n]
        self.i += n
        self.i = min(self.i, len(self.s))
        return s

    def write(self, s):
        self.s += s

s = 'hello\nwo\1\200\trld\n'

f = MyStringIO(s)
f.write('peas\n')
f

for i, line in enumerate(f):
    print(i, repr(line))

f = MyStringIO(s)
f.write('peas\n')
f

for i, line in enumerate(pass_good_characters(f)):
    print(i, repr(line))

# str.isprintable() for many characters above ASCII.
for i in range(2 * N_ASCII_CHARACTERS):
    c = chr(i)
    print(i, repr(c), c.isprintable())

string.digits

string.ascii_letters

string.punctuation

s = set('\t' + ' ' + string.digits + string.ascii_letters + string.punctuation)
len(s)

