#importing modules
import math
from __future__ import division

#Variable declaration
A=12000;     #investment of A and B(Rs)
P=1800;     #profit(Rs)
Pa=750;     #profit of A(Rs)

#Calculation
ratio=Pa/(P-Pa);     #ratio of investments
Ia=Pa*A/P;      #investment of A(Rs)

#Result
print "investment of A is",Ia,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
Ca=10000;    #capital of A(Rs)
na=12;    #number of years
nb=4;     #number of years
Cb=5000;    #capital of B(Rs)
P=2000;    #total profit(Rs)

#Calculation
a=Ca*na;
b=Cb*(na-nb);
r=a/b;     #ratio of profits
Pa=a*P/(a+b);     #profit share of A(Rs)

#Result
print "profit share of A is",Pa,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
Ia=380;     #investment of A(Rs)
Ib=400;     #investment of B(Rs)
Ic=420;     #investment of C(Rs)
P=180;     #net profit(Rs)

#Calculation
I=Ia+Ib+Ic;     #total investment(Rs)
Pa=Ia*P/I;     #profit of A(Rs)
Pb=Ib*P/I;     #profit of B(Rs)
Pc=Ic*P/I;     #profit of C(Rs)

#Result
print "profit of A is",Pa,"Rs"
print "profit of B is",Pb,"Rs"
print "profit of C is",Pc,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
Ca=320*4;    #contribution of A for 4 months(Rs)
Cb=510*3;    #contribution of B for 3 months(Rs)
Cc=270*5;    #contribution on C for 5 months(Rs)
P=208;     #net profit(Rs)

#Calculation
C=Ca+Cb+Cc;     #contribution of A,B,C(Rs)
Pa=Ca*P/C;     #profit of A(Rs)
Pb=Cb*P/C;     #profit of B(Rs)
Pc=Cc*P/C;     #profit of C(Rs)

#Result
print "profit of A is",Pa,"Rs"
print "profit of B is",Pb,"Rs"
print "profit of C is",Pc,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
C=8200;    #total capital(Rs)
a=1000;
b=2000;
Cb=(C-(b+a+a))/3;      #capital of B(Rs)
P=2460;     #total profit(Rs)

#Calculation
Pb=Cb*P/C;    #profit of B(Rs)

#Result
print "profit of B is",Pb,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
m=50;     #amount with each person(Rs)
mA=20;    #amount left with A(Rs)
mB=30;    #amount left with B(Rs)
mC=40;    #amount left with C(Rs)

#Calculation
Ms=m*3;     #total amount started with(Rs)
Mr=mA+mB+mC;    #total amount ending with(Rs)
M=Mr/3;        #amount each person must have(Rs)
Cm=mC-M;      #C must pay(Rs)

#Result
print "C must pay Rs",Cm,"to A"

#importing modules
import math
from __future__ import division

#Variable declaration
a=1290;    #amount(Rs)
A=3;    #value of A in the ratio A:B
B1=2;   #value of B in the ratio A:B 
B2=7;   #value of B in the ratio B:C
C1=4;   #value of C in the ratio B:C

#Calculation
#inorder to make the value of B in the ratios same, multiply by B2
A=A*B2;
B=B1*B2;
C=C1*B1;     #new values of the ratio A:B:C
Sc=C*a/(A+B+C);    #share of C(Rs)

#Result
print "share of C is",Sc,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
A=390;    #amount A receives(Rs)
Ap=10*12;    #profit for A(Rs)
Ia=3000;    #investment of A(Rs)
Ib=4000;    #investment of B(Rs)

#Calculation
Ba=A-Ap;    #balance profit of A(Rs)
Bb=Ba*Ib/Ia;     #balance profit of B(Rs)

#Result
print "balance profit of B is",Bb,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
BplusC=100;     #amount with B and C(Rs)
AplusC=150;      #amount with A and C(Rs)

#Calculation
B=AplusC-BplusC;     #amount with B(Rs)
ABC=AplusC+B;        #total amount of money(Rs)

#Result
print "total amount of money is",ABC,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
A=760;    #amount(Rs)
Ar=4;     #part of A in ratio
Br=5;     #part of B in ratio
At=3;     #time for A(months)
AT=10-3;   #ending time(months)

#Calculation
MEI_A=(Ar*At)+(AT*Ar*At/Ar);    #monthly equivalent investment of A
MEI_B=(Br*At)+(AT*Br*Ar/Br);    #monthly equivalent investment of B
PA=MEI_A*A/(MEI_A+MEI_B);     #profit share of A(Rs)
PB=MEI_B*A/(MEI_A+MEI_B);     #profit share of B(Rs)

#Result
print "profit share of A is",PA,"Rs"
print "profit share of B is",PB,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
Fd_A=27;     #floppy disks of A
Ad=19;    #number of days for A
Fd_B=21;  #floppy disks of B
Bd=17;    #number of days for B
Fd_C=24;  #floppy disks of C
Cd=23;    #number of days for C
am=23700;    #amount(Rs)

#Calculation
A=Fd_A*Ad;    #A's floppy days
B=Fd_B*Bd;    #B's floppy days
C=Fd_C*Cd;    #C's floppy days
PA=A*am/(A+B+C);    #payment for rent by A(Rs)
PB=B*am/(A+B+C);    #payment for rent by B(Rs)
PC=C*am/(A+B+C);    #payment for rent by C(Rs)

#Result
print "payment for rent by A is",PA,"Rs"
print "payment for rent by B is",PB,"Rs"
print "payment for rent by C is",PC,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
S=1000;     #share of A(Rs)
IA=8;     #investment of A(months)
IB=12;     #investment of B(months)
PA=PB=1;  

#Calculation
CA=(PA/PB)*S*(IB/(IA/2));        #capital of A(Rs)

#Result
print "capital of A is",CA,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
Sa=4000;     #A's share(s)
Sb=5000;     #B's share(Rs)
Sc=6000;     #C's share(Rs)
r=1000;    #ratio fraction
p=75/100;  #profit percentage(%)
P=25/100;
A=100;    #A gets less(Rs)

#Calculation
TP=((Sa/r)+(Sb/r)+(Sc/r))/p;     #total profit
As=(Sa/r)+(P*TP);         #share of A
Bs=Sb/r;         #share of B
Cs=Sc/r;         #share of C
x=A/((Bs+Cs)-As);
Tp=TP*x;         #total profit(Rs)

#Result
print "total profit is",Tp,"Rs"
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
AP40=1250;     #A's 40% profit(Rs)
BP40=850;      #B's 40% profit(Rs)
A=30;    #amount received more(Rs)
p=60/100;   #distributed profit(%)

#Calculation
R=(AP40+BP40)/(AP40-BP40);      #applying componendo dividendo
P=R*A/(1-p);         #total profit(Rs)

#Result
print "total profit is",P,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
CA=1/6;       #capital of A
CB=1/3;       #capital of B
P=2300;     #total profit(Rs)

#Calculation
CC=1-CA-CB;    #capital of C
PA=CA*CA*12;   #A's profit
PB=CB*CB*12;   #B's profit
PC=CC*12;      #C's profit
SA=PA*P/(PA+PB+PC);       #share of A(Rs)

#Result
print "share of A is",SA,"Rs"

