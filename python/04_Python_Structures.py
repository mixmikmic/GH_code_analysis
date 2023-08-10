print "Hello World!"

x=42
print x+10
print x/4
x="42"
print x+10
print x+"10"

x=[1, 2, 3]
y=[4,5, 6]
print x
print x*2
print x+y

print range(10)
print range(20, 50, 3)
print []

x=range(10)
print x
print "First value", x[0]
print "Last value", x[-1]
print "Fourth to sixth values", x[3:5]

x=[1,2,3,4,5]
x[2]=8
print x

print "Testing append"
x.append(6)
print x
x.append([7,8])
print x

print "testing extend"
x=[1,2,3,4,5]
#x.extend(6)
#print x
x.extend([7,8])
print x

print "testing insert"
x=[1,2,3,4,5]
x.insert(3, "in")
print x

x=range(1,11,1)
print x
x_2=[]
for i in x:
    i_2=i*i
    x_2.append(i_2)
print x_2

x=range(1,11,1)
print x
x_2=[i*i for i in x]
print x_2

x={}
x['answer']=42
print x['answer']

AbMag={'U':5.61, 'B':5.48, 'V':4.83, 'R':4.42, 'I':4.08}
print AbMag['U']
print AbMag.items()

def GeoSum(r):
    powers=range(1,11,1) #set up a list for the exponents 1 to 10
    terms=[(1./(r**x)) for x in powers] #calculate each term in the series
    return sum(terms) #return the sum of the list

TermValue=2
print GeoSum(TermValue), (1.)/(TermValue-1)

class SampleClass:
    def __init__(self, value): #run on initial setup of the class, provide a value
       self.value = value
       self.square = value**2
    
    def powerraise(self, powerval): #only run when we call it, provide powerval
        self.powerval=powerval
        self.raisedpower=self.value**powerval

MyNum=SampleClass(3)
print MyNum.value
print MyNum.square
MyNum.powerraise(4)
print MyNum.powerval
print MyNum.raisedpower
print MyNum.value,'^',MyNum.powerval,'=',MyNum.raisedpower

