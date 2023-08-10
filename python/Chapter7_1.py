#importing modules
import math
from __future__ import division

#Variable declaration
a=0.75;
b=5;
c=8;

#Calculation
x=a*c/b;

#Result
print "result is",x

#importing modules
import math
from __future__ import division

#Variable declaration
a=0.32;
b=0.02;

#Calculation
x2=a*b;
x=math.sqrt(a*b);    #mean proportion

#Result
print "mean proportion is",x

#importing modules
import math
from __future__ import division

#Variable declaration
a=16;
b=24;

#Calculation
x=(b**2)/a;     #third proportional

#Result
print "third proportional is",x

#importing modules
import math
from __future__ import division

#Variable declaration
a=16;
b=4;

#Calculation
x=(b**2)/a;     #third proportional

#Result
print "third proportional is",x

#importing modules
import math
from __future__ import division
from fractions import Fraction

#Variable declaration
a=3/48;
b=1/12;

#Calculation
p=a/b;      #part

#Result
print "part is",Fraction(p)

#importing modules
import math
from __future__ import division
from fractions import Fraction

#Variable declaration
a=7;
b=8;

#Calculation
p=a/b;      #part

#Result
print "part is",Fraction(p)

#importing modules
import math
from __future__ import division

#Variable declaration
a=4/3;     #ratio of number of boys and girls
b=480;    #number of boys

#Calculation
x=b/a;    #number of girls

#Result
print "number of girls is",x

#importing modules
import math
from __future__ import division

#Variable declaration
a=4;     #first ratio term
b=5;     #second ratio term
c=72;    #total amount(Rs)

#Calculation
fp=a*c/(a+b);   #first part(Rs)
sp=b*c/(a+b);   #second part(Rs)

#Result
print "first part is",fp,"Rs"
print "second part is",sp,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
a=2;     #first ratio term
b=5;     #second ratio term
c=350;   #number of students

#Calculation
x=b*c/(a+b);   #number of girls

#Result
print "number of girls is",x

#importing modules
import math
from __future__ import division

#Variable declaration
a=3;   #first ratio term
b=7;   #second ratio term
lcm=210;   #lcm of two numbers

#Calculation
x=lcm/(a*b);   #factor
n1=a*x;   #first number
n2=b*x;   #second number

#Result
print "the numbers are",n1,"and",n2

#importing modules
import math
from __future__ import division

#Variable declaration
a=3;   #first ratio term
b=7;   #second ratio term
c=33;   #antecedent

#Calculation
x=b*c/a;   #consequent

#Result
print "consequent is",x

#importing modules
import math
from __future__ import division

#Variable declaration
aold=3;   #first ratio term
bold=5;   #second ratio term
c=4;   #increment
anew=2;   #first term of new ratio
bnew=3;   #second term of new ratio

#Calculation
Nr=(aold*c)-(anew*c);
Dr=(bold*anew)-(aold*bnew);
x=Nr/Dr;     #factor
n1=x*aold;
n2=x*bold;  

#Result
print "the numbers are",n1,"and",n2

#importing modules
import math
from __future__ import division

#Variable declaration
aold=12;   #first ratio term
bold=13;   #second ratio term
c=20;   #decrement
anew=2;   #first term of new ratio
bnew=3;   #second term of new ratio

#Calculation
Nr=(bnew*c)-(anew*c);
Dr=(bnew*aold)-(anew*bold);
x=Nr/Dr;     #factor
n1=x*aold;
n2=x*bold;  

#Result
print "the numbers are",n1,"and",n2

#importing modules
import math
from __future__ import division

#Variable declaration
aold=3;   #first ratio term
bold=7;   #second ratio term
anew=2;   #first term of new ratio
bnew=5;   #second term of new ratio

#Calculation
Nr=(aold*bnew)-(anew*bold);
Dr=bnew-anew;
x=Nr/Dr;    #decrement

#Result
print "decrement is",round(x,2)
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
aold=1;   #first ratio term
bold=2;   #second ratio term
anew=1;   #first term of new ratio
bnew=3;   #second term of new ratio
g=2;   #number of gents left
l=2;   #number of ladies left

#Calculation
Nr=(g*bnew)-(anew*l);
Dr=bnew-(anew*l);
x=Nr/Dr;    #factor
n=x*(aold+bold);     #number of people originally present

#Result
print "number of people originally present is",n

#importing modules
import math
from __future__ import division

#Variable declaration
c1=1;     #coin value 1 rupee
c2=0.5;   #coin value 50 paise
c3=0.25;  #coin value 25 paise
t=35;     #total amount(Rs)

#Calculation
n=t/(c1+c2+c3);     #number of each type of coin(coins)

#Result
print "number of each type of coin is",n,"coins"

#importing modules
import math
from __future__ import division

#Variable declaration
a=3/7;
b=1/5;
c=7/15;

#Calculation
x=a*b/c;      #fraction of same ratio

#Result
print "fraction of same ratio is",x

#importing modules
import math
from __future__ import division

#Variable declaration
s=90;    #sum of ages of A, B, C(years)
n=6;     #number of years ago
a=1;     #ratio term for age of A
b=2;     #ratio term for age of B
c=3;     #ratio term for age of C

#Calculation
age=s-(n*c);    #sum of ages of A, B, C 6 years ago(years)
C=c*age/(a+b+c);    #age of C 6 years ago(years)
Cp=C+a+b+c;         #present age of C(years)

#Result
print "present age of C is",Cp,"years"

#importing modules
import math
from __future__ import division

#Variable declaration
s=532;     #sum of squares of 3 numbers
a=3;      #first term of ratio
b=2;      #second term of ratio

#Calculation
n1=a*a;   #first number
n2=a*b;   #second number
n3=b*b;   #third number
x2=s/((n1**2)+(n2**2)+(n3**2));
x=math.sqrt(x2);     #factor
N2=n2*x;     #second number

#Result
print "second number is",N2

#importing modules
import math
from __future__ import division

#Variable declaration
l1=60;     #length of wire(m)
w1=80;     #weight of wire(kg)
l2=141;    #length of another wire(m)

#Calculation
x=w1*l2/l1;    #weight of wire(kg)

#Result
print "weight of wire is",x,"kg"

#importing modules
import math
from __future__ import division

#Variable declaration
bc=1;    #first term of ratio
ac=2;    #second term of ratio
ab=3;    #third term of ratio

#Calculation
#abyb=ac/bc
a=ac;
b=bc;
result=(a**2)/(b**2);     #value of a/bc : b/ca

#Result
print "value of a/bc : b/ca is",int(result)

#importing modules
import math
from __future__ import division

#Variable declaration
v1=1;     #one rupee coin
v2=1/2;   #50 paise coin
v3=1/4;   #25 paise coin
v4=1/10;   #10 paise coin
n1=1;   #number of 1 rupee coins
n2=3;   #number of 50 paise coins
n3=5;   #number of 25 paise coins
n4=7;   #number of 10 paise coins
s=22.25;    #total amount(Rs)

#Calculation
a=(v1*n1)+(v2*n2)+(v3*n3)+(v4*n4);
x=s/a;     #factor
N1=x*n1;   #number of 1 rupee coins
N2=x*n2;   #number of 50 paise coins
N3=x*n3;   #number of 25 paise coins
N4=x*n4;   #number of 10 paise coins

#Result
print "number of 1 rupee coins is",N1
print "number of 50 paise coins is",N2
print "number of 25 paise coins is",N3
print "number of 10 paise coins is",N4

#importing modules
import math
from __future__ import division
from fractions import gcd

#Variable declaration
a=1/3;
b=1/4;
c=1/5;
d=1/6;

#Calculation
l1=(1/a)*(1/b)/gcd(1/a,1/b);
l2=(1/c)*(1/d)/gcd(1/c,1/d);
l=l1*l2/gcd(l1,l2);           #lcm of 3,4,5,6
A=a*l;
B=b*l;
C=c*l;
D=d*l;
p=A+B+C+D;

#Result
print "the person should have atleast",p,"pens"

