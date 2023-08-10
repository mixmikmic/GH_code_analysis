#importing modules
import math
from __future__ import division

#Variable declaration
t1=20;      #number of days A can do a job
t2=30;      #number of days B can do a job

#Calculation
T=t1*t2/(t1+t2);     #number of days for A and B to do the job together

#Result
print "A and B can do the job together in",T,"days"

#importing modules
import math
from __future__ import division

#Variable declaration
t1=28;     #number of days B can do a job
T=12;      #number of days for A and B to do the job together

#Calculation
t2=T*t1/(t1-T);     #number of days A can do a job

#Result
print "A can do the job in",t2,"days"

#importing modules
import math
from __future__ import division
from fractions import gcd

#Variable declaration
T1=12;      #work done by A and B together(days)
T2=15;      #work done by C and B together(days)
T3=20;      #work done by A and C together(days)

#Calculation
l1=T1*T2/gcd(T1,T2);     #lcm of T1 and T2
L=l1*T3/gcd(l1,T3);     #lcm of the given 3 numbers
a=L/T1;
b=L/T2;
c=L/T3;
T=2*L/(a+b+c);         #work done by A, B and C together(days)
Ta=2*L/(a-b+c);         #work done by A alone(days)
Tb=2*L/(a+b-c);         #work done by B alone(days)
Tc=2*L/(-a+b+c);         #work done by C alone(days)

#Result
print "work done by A, B and C together is",T,"days"
print "work done by A alone is",Ta,"days"
print "work done by B alone is",Tb,"days"
print "work done by C alone is",Tc,"days"

#importing modules
import math
from __future__ import division

#Variable declaration
Ta=10;     #alone time of A(days)
TB=3;      #number of days B worked(days)
Tb=15;     #alone time of B(days)

#Calculation
T=Ta*(1-(TB/Tb));        #total work finished(days)

#Result
print "total work is finished in",T,"days"

#importing modules
import math
from __future__ import division
from fractions import gcd

#Variable declaration
Ta=20;     #alone time of A(days)
Tb=30;     #alone time of B(days)
a=5;       #number of days A leaves(days)

#Calculation
L=Ta*Tb/gcd(Ta,Tb);     #lcm of T1 and T2
l1=L/Ta;
l2=L/Tb;
T=(L+(l1*a))/(l1+l2);     #total work finished(days)

#Result
print "total work is finished in",T,"days"

#importing modules
import math
from __future__ import division
from fractions import gcd

#Variable declaration
Ta=10;     #alone time of A(days)
Tb=12;     #alone time of B(days)
a=6;       #number of days after A started(days)

#Calculation
L=T1*T2/gcd(T1,T2);     #lcm of T1 and T2
l1=L/Ta;
l2=L/Tb;
T=(L+(l1*a))/(l1+l2);     #total work finished(days)
TB=T-a;         #number of days B worked for(days)

#Result
print "total work is finished in",T,"days"
print "B worked for",TB,"days"

#importing modules
import math
from __future__ import division

#Variable declaration
M1=1;     #number of men
E1=1;     #efficiency of anil
D1=8;     #number of days worked(days)
E=60/100;    #efficiency of rakesh(%)
P1=1/3;      #part of work done by anil
M2=1;     #number of men

#Calculation
E2=(M1*M2)+(E1*E);      #efficiency
P2=1-P1;     #part of work done by rakesh
D2=M1*E1*D1*P2/(P1*E2);     #number of days to complete the job(days)

#Result
print "number of days to complete the job is",D2,"days"

#importing modules
import math
from __future__ import division
from fractions import Fraction

#Variable declaration
T1=5;      #work done by A and B together(days)
T=1;        #total job done
eB=1/3;     #efficiency of B

#Calculation
t=T-(T/T1);   
tA=T1*(1/t);     #time taken for A to complete the job(days)
x=tA-int(tA);

#Result
print "time taken for A to complete the job is",int(tA),Fraction(x),"days"

#importing modules
import math
from __future__ import division

#Variable declaration
M1=1;     #number of men
D1=12;     #number of days worked(days)
M2=1;     #number of men
W1=3/4;    #work done by A
W2=1/8;    #part of work done

#Calculation
d=M1*D1*W2/(W1*M2);     #number of days for A(days)

#Result
print "A can finish in",d,"days"

#importing modules
import math
from __future__ import division
from fractions import gcd

#Variable declaration
Ta=10;     #alone time of A(days)
Tb=12;     #alone time of B(days)
Tc=15;     #alone time of C(days)
a=2;       #number of days after A started(days)
b=3;       #number of days before the work finished(days)

#Calculation
L1=Ta*Tb/gcd(Ta,Tb);     #lcm of Ta and Tb
L=L1*Tc/gcd(L1,Tc);      #lcm of all the three
l1=L/Ta;
l2=L/Tb;
l3=L/Tc;
r=1-a/Ta;
T=((r*L)+Tc)/(l2+l3);     #total work finished(days)
TB=T-a;         #number of days B worked for(days)

#Result
print "total work is finished in",T,"days"

#importing modules
import math
from __future__ import division
from fractions import gcd

#Variable declaration
T1=15;     #alone time of A(days)
T2=10;     #alone time of B(days)
a=5;       #number of days for B to finish(days)

#Calculation
L=T1*T2/gcd(T1,T2);     #lcm of T1 and T2
l1=L/T1;
l2=L/T2;
x=(L-(l2*a))/(l1+l2);     #A left after days(days)

#Result
print "A left after",x,"days"

#importing modules
import math
from __future__ import division

#Variable declaration
T1=12;     #alone time of A(days)
T2=15;     #alone time of B(days)
T=2;       #number of days they work together(days)
t=5;       #number of days work is completed(days)

#Calculation
Ta=(1+T+t)*(1/T1);      #A's amount of work
Tb=T*(1/T2);       #B's amount of work
T3=1-(Ta+Tb);
t3=t/T3;         #C can do it in(days)

#Result
print "C can do it alone in",t3,"days"

#importing modules
import math
from __future__ import division
from fractions import gcd

#Variable declaration
p=25;     #amount paid(Rs)
A=32;     #number of days for A(days)
B=20;     #number of days for B(days)
C=12;     #number of days for C(days)
D=24;     #number of days for B(days)

#Calculation
l1=A*B/gcd(A,B);     #lcm of A and B
l2=C*D/gcd(C,D);     #lcm of C and D
L=l1*l2/gcd(l1,l2);   #LCM of A,B,C and D
a=L/A;
b=L/B;
c1=(1/C)-(1/B);
c=L*c1;
d=L/D;
Cs=c*p/(a+b+c+d);     #amount received by C(Rs)

#Result
print "amount received by C is",round(Cs,2),"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
tA=5;     #work for A alone(days)
tB=7;     #work for B alone(days)

#Calculation
p=round(tA*tB/(tA+tB));    #nearest integer value
Ta=((tA*tB)+(p*(tA-tB)))/tA;     #time taken to finish the job if A starts the work(days)
Tb=((tA*tB)-(p*(tA-tB)))/tB;     #time taken to finish the job if B starts the work(days)

#Result
print "time taken to finish the job if A starts the work is",Ta,"days"
print "time taken to finish the job if B starts the work is",round(Tb,2),"days"

#importing modules
import math
from __future__ import division

#Variable declaration
tA=20;     #number of days for A(days)
tB=30;     #number of days for B(days)
tC=60;     #number of days for C(days)
da=3;      #help done by A
db=1;
dc=1;

#Calculation
t=(da/tA)+(db/tB)+(dc/tC);
T=da/t;       #job is finished in(days)

#Result
print "job is finished in",T,"days"

#importing modules
import math
from __future__ import division

#Variable declaration
K=3;
l=80;     #number of days

#Calculation
T=K*l/((K**2)-1);     #job is finished in(days)

#Result
print "job is finished in",T,"days"

#importing modules
import math
from __future__ import division

#Variable declaration
tA=20;     #work by skilled men(days)
tB=30;     #work by boys(days)

#Calculation
T=tA*tB/(tA+tB);       #time taken if they work together(days) 

#Result
print "time taken if they work together is",T,"days"

