#importing modules
import math
from __future__ import division

#Variable declaration
C1=125;    #CP of 1st article(Rs)
S1=96;     #SP of 1st article(Rs)
C2=112;    #CP of 2nd article(Rs)
S2=132;     #SP of 2nd article(Rs)
C3=120;    #CP of 3rd article(Rs)
S3=90;     #SP of 3rd article(Rs)
C4=80;    #CP of 4th article(Rs)
S4=100;     #SP of 4th article(Rs)
C5=90;    #CP of 5th article(Rs)
G5=10;    #gain(%)
L6=25;    #loss(%)
C6=20;    #CP of 6th article(Rs)
S7=84;     #SP of 7th article(Rs)
G7=20;    #gain(%)

#Calculation
x1=S1-C1;
L1=-x1;       #loss in 1st case(Rs)
x2=S2-C2;
G2=x2;       #gain in 2nd case
x3=S3-C3;
L3=-x3*100/C3;    #loss in 3rd case(%)
x4=S4-C4;
G4=x4*100/C4;    #gain in 4th case(%)
S5=(100+G5)*C5/100;    #SP in 5th case(Rs)
S6=(100-L6)*C6/100;    #SP in 6th case(Rs)
C7=S7/(1+(G7*0.01));    #CP in 7th case(Rs)  

#Result
print "loss in 1st case is",L1,"Rs"
print "gain in 2nd case is",G2,"Rs"
print "loss in 3rd case is",L3,"%"
print "gain in 4th case is",G4,"%"
print "SP in 5th case is",S5,"Rs"
print "SP in 6th case is",S6,"Rs"
print "CP in 7th case is",C7,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
S1=450;      #SP of 1st article(Rs)
x1=25;     #loss(%)
x2=25;     #gain(%)

#Calculation
S2=S1*(100+x2)/(100-x1);      #SP of 2nd article(Rs)

#Result
print "SP of 2nd article is",S2,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
n=25;     #number of articles on CP
N=20;     #number of articles on SP

#Calculation
x=(n-N)*100/N;     #gain in transaction(%)

#Result
print "gain in transaction is",x,"%"

#importing modules
import math
from __future__ import division

#Variable declaration
S1=1/36;      #SP of 1st article(Rs)
x1=4;     #loss(%)
x2=8;     #gain(%)

#Calculation
S2=S1*(100+x2)/(100-x1);      #SP of 2nd article(Rs)
n=1/S2;     #number of oranges per rupee
 
#Result
print "number of oranges per rupee is",n

#importing modules
import math
from __future__ import division

#Variable declaration
q1=2;      #quantity(kg)
q2=10;     #quantity(kg)
C=600;     #cost price(Rs)

#Calculation
S=C/(q1+q2);     #sale rate of rice(Rs)

#Result
print "sale rate of rice is Rs",S,"per kg"

#importing modules
import math
from __future__ import division

#Variable declaration
S1=455;      #SP of 1st article(Rs)
S2=555;      #SP of 2nd article(Rs)
x1=9;     #loss(%)

#Calculation
x2=(S2*(100-x1)/S1)-100;     #gain(%)

#Result
print "gain is",x2,"%"

#importing modules
import math
from __future__ import division

#Variable declaration
p=20;      #profit on SP(%)

#Calculation
rp=p*100/(100-p);     #real profit(%)

#Result
print "real profit is",rp,"%"

#importing modules
import math
from __future__ import division

#Variable declaration
l=-10;      #loss on SP(%)

#Calculation
rl=-l*100/(100-l);     #real loss(%)

#Result
print "real loss is",round(rl,2),"%"

#importing modules
import math
from __future__ import division

#Variable declaration
d=10;    #discount(%)
g=26;    #profit(%)
C=1;   #assume

#Calculation
x=C*(100+g)/(100-d);     
M=(x-1)*100;     #percentage increase of MP(%)

#Result
print "percentage increase of MP is",M,"%"

#importing modules
import math
from __future__ import division

#Variable declaration
S1=30;      #SP of 1st article(Rs)
x1=20;     #gain(%)
d=10/100;   #discount

#Calculation
S2=S1*(1-d);      #SP of 2nd article(Rs)
x2=(S2*(100+x1)/S1)-100;     #gain(%)

#Result
print "gain is",x2,"%"

#importing modules
import math
from __future__ import division

#Variable declaration
x=40;     #gain(%)
S=1/10;   #SP of article(Rs)

#Calculation
y=(x/100)+1;
C=S/y;      #CP of article(Rs)
n=1/C;     #number of oranges per rupee

#Result
print "number of oranges per rupee is",n

#importing modules
import math
from __future__ import division

#Variable declaration
SP=1/4;    #SP of article
CP=1/6;    #CP of article
tg=26;    #total gain(Rs)

#Calculation
g=(SP-CP)*100/CP;      #gain(%)
x=tg/(SP-CP);        #number of oranges

#Result
print "gain is",g,"%"
print "number of oranges is",x

#importing modules
import math
from __future__ import division

#Variable declaration
x=0;
Tw=1000;      #true weight(kg)
Fw=900;     #false weight(kg)

#Calculation
G=(Tw*(100+x)/Fw)-100;      #gain(%)

#Result
print "gain is",round(G,2),"%"

#importing modules
import math
from __future__ import division

#Variable declaration
Ts=100;   #true scale(cm)
G=15;    #gain(%)
x=10;    #loss(%)

#Calculation
l=Ts*(100-x)/(100+G);     #false scale length(cm)

#Result
print "false scale length is",round(l,2),"cm"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
x1=25;    #gain(%)
x2=-20;    #loss(%)

#Calculation
x=2*(100+x1)*(100+x2);
y=100+x1+100+x2;
g=(x/y)-1;        #gain(%)

#Result
print "gain is",round(g,2),"%"
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
x=12;     #loss or gain(%)

#Calculation
l=-(x/10)**2;      #overall loss(%)

#Result
print "overall loss is",l,"%"

#importing modules
import math
from __future__ import division

#Variable declaration
C1=10/100;     #loss
C2=15/100;     #profit
C=560;     #cost(Rs)

#Calculation
C1=C1*C/(C1+C2);     #cost price of 1st watch(Rs)
C2=C-C1;      #cost price of 2nd watch(Rs)

#Result
print "cost price of 1st watch is",C1,"Rs"
print "cost price of 2nd watch is",C2,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
C1=100;     #CP of book(Rs)
p1=20/100;     #profit
C2=80;     #assume 20% less
p2=25/100;   #profit in 2nd case
d=18;     #given difference(Rs)

#Calculation
S1=C1*(1+p1);     #selling price of book(Rs)
S2=C2*(1+p2);      #selling price in 2nd case(Rs)
S=S1-S2;     #difference when CP=100(Rs)
CP=d*C1/S;   #cost price of book(Rs)

#Result
print "cost price of book is",CP,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
d=16;     #discount(%)
M_S=80;    #cost(Rs)

#Calculation
S=(100-d)*M_S/d;      #cost price of the book(Rs)

#Result
print "cost price of the book is",S,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
n=30;     #number of articles on CP
N=27;     #number of articles on SP

#Calculation
x=(n-N)*100/N;     #gain in transaction(%)

#Result
print "gain in transaction is",round(x,2),"%"

#importing modules
import math
from __future__ import division

#Variable declaration
x=80;    #quantity(kg)
y=20;    #quantity(kg)
C=88;    #cost price(Rs)

#Calculation
S=C/(x+y);     #selling price of salt(Rs)

#Result
print "selling price of salt is",S*100,"paise per kg"

#importing modules
import math
from __future__ import division

#Variable declaration
l=10;    #loss(%)
f=3/4;   #fraction 

#Calculation
SP=(100-l)/f;      #ratio of SP to CP(%)
p=SP-100;     #profit on the article(%)

#Result
print "profit on the article is",p,"%"

#importing modules
import math
from __future__ import division

#Variable declaration
N=66;    #length of cloth(m)
x=22;    #gain in length(m)

#Calculation
g=x*100/(N-x);     #gain(%)

#Result
print "gain is",g,"%"

#importing modules
import math
from __future__ import division

#Variable declaration
N=66;    #length of cloth(m)
Y=22;    #gain in length(m)

#Calculation
g=Y*100/N;     #gain(%)

#Result
print "gain is",round(g,2),"%"

#importing modules
import math
from __future__ import division

#Variable declaration
x=-22;     #loss in length(m)
N=66;      #length of cloth(m)

#Calculation
l=-x*100/(N-x);     #loss(%)

#Result
print "loss is",l,"%"

