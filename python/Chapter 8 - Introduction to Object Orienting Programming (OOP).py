class Square:
    def __init__(self):
        self.side = 1

Bob = Square() # Bob is an instance of Square.
Bob.side #Let’s see the value of side

Bob.side = 5 #Assing a new value to side
Bob.side #Let’s see the new value of side

Krusty = Square()
Krusty.side    

class Square:
    def __init__(self):
        self.side=1
        
Bob = Square() # Bob is an instance of Square.
Bob.side #Let's see the value of side

Bob.side = 5 #Assing a new value to side
Bob.side #Let's see the new value of side

Krusty = Square()
Krusty.side

Square.side

Crab = Square()
Crab.side    

class Square:
    count = 0
    def __init__(self):
        Square.count += 1
        print("Object created successfully")

Bob = Square()

Patrick = Square()

Square.count

class Sequence:
    transcription_table = {'A':'U', 'T':'A', 'C':'G' , 'G':'C'}
    def __init__(self, seqstring):
        self.seqstring = seqstring.upper()
    def transcription(self):
        tt = ""
        for letter in self.seqstring:
            if letter in 'ATCG':
                tt += self.transcription_table[letter]
        return tt

dangerous_virus = Sequence('atggagagccttgttcttggtgtcaa')
dangerous_virus.seqstring

harmless_virus = Sequence('aatgctactactattagtagaattgatgcca')
harmless_virus.seqstring

dangerous_virus.transcription()

class Sequence:
    transcription_table = {'A':'U', 'T':'A', 'C':'G' , 'G':'C'}
    enz_dict = {'EcoRI':'GAATTC', 'EcoRV':'GATATC'}
    def __init__(self, seqstring):
        self.seqstring = seqstring.upper()
    def restriction(self, enz):
        try:
            enz_target = Sequence.enz_dict[enz]
            return self.seqstring.count(enz_target)
        except KeyError:
            return 0
    def transcription(self):
        tt = ""
        for letter in self.seqstring:
            if letter in 'ATCG':
                tt += self.transcription_table[letter]
        return tt

other_virus = Sequence('atgatatcggagaggatatcggtgtcaa')
other_virus.restriction('EcoRV')

class Mammal():
    """Docstring with class description"""
    # Properties here
    # Methods here

class Orca(Mammal):
    """Docstring with class description"""
    # Properties here
    # Methods here

class Plasmid(Sequence):
    ab_res_dict = {'Tet':'ctagcat', 'Amp':'CACTACTG'}
    def __init__(self, seqstring):
        Sequence.__init__(self, seqstring)
    def ab_res(self, ab):
        if self.ab_res_dict[ab] in self.seqstring:
            return True
        return False

get_ipython().system('conda install biopython -y')

from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
first_seq = Seq('GCTATGCAGC', IUPAC.unambiguous_dna)
first_seq

first_seq.complement()

first_seq.tostring()

first_seq[:10] # slice a sequence

len(first_seq) # get the length of the sequence

first_seq[0] # get one character

first_seq

AnotherSeq=first_seq.tomutable()
AnotherSeq.extend("TTTTTTT")
print(AnotherSeq)

AnotherSeq.pop()

AnotherSeq.pop()

print(AnotherSeq)

class Sequence:
    transcription_table = {'A':'U', 'T':'A', 'C':'G' , 'G':'C'}
    enz_dict = {'EcoRI':'GAATTC', 'EcoRV':'GATATC'}
    def __init__(self, seqstring):
        self.seqstring = seqstring.upper()
    def __len__(self):
        return len(self.seqstring)
    def restriction(self, enz):
        try:
            enz_target = Sequence.enz_dict[enz]
            return self.seqstring.count(enz_target)
        except KeyError:
            return 0
    def transcription(self):
        tt = ""
        for letter in self.seqstring:
            if letter in 'ATCG':
                tt += self.transcription_table[letter]
        return tt
    
M13 = Sequence("ACGACTCTCGACGGCATCCACCCTCTCTGAGA")
len(M13)

class Straight:
    def __init__(self, data):
        self.data = data
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.index == len(self.data):
            raise StopIteration
        answer = self.data[self.index]
        self.index += 1
        return answer

class Reverse:
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    def __iter__(self):
        return self
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]
    
a = Straight("123")
for i in a:
    print(i) 

b = Reverse("123")
for i in b:
    print(i)

class Sequence:
    transcription_table = {'A':'U', 'T':'A', 'C':'G', 'G':'C'}
    comp_table = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    def __init__(self, seqstring):
        self.seqstring = seqstring.upper()
    def restriction(self, enz):
        enz_dict = {'EcoRI':'ACTGG', 'EcoRV':'AGTGC'}
        try:
            target = enz_dict[enz]
        except KeyError:
            raise ValueError('No such enzime in out enzime DB')
        return self.seqstring.count(target)
    def __getitem__(self,index):
        return self.seqstring[index]
    def __getslice__(self, low, high):
        return self.seqstring[low:high]
    def __len__(self):
        return len(self.seqstring)
    def __str__(self):
        if len(self.seqstring) >= 28:
            return '{0}...{1}'.format(self.seqstring[:25], 
                                      self.seqstring[-3:])
        else:
            return self.seqstring
    def transcription(self):
        tt = ''
        for x in self.seqstring:
            if x in 'ATCG':
                tt += self.transcription_table[x]
        return tt
    def complement(self):
        tt = ''
        for x in self.seqstring:
            if x in 'ATCG':
                tt += self.comp_table[x]
        return tt


class Zdic(dict):
    """ A dictionary-like object that return 0 when a user
    request a non-existent key.
    """

    def __missing__(self,x):
        return 0

a = Zdic()
a['blue'] = 'azul'
a['red']

class TestClass:
    """A class with a "private" method (b)"""
    def a(self):
        pass
    def __b(self):
        # mangled to _TestClass__b
        pass

my_object = TestClass()
print(my_object.a())
my_object.__b()

my_object._TestClass__b()

