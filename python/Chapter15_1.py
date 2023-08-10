#importing modules
import math
from __future__ import division

#Variable declaration
P=4000;    #principal(Rs)
R=5;       #rate(%)
T=2;       #time(yrs)
n=1;

#Calculation
A=P*(1+(R/(100*n)))**(n*T);     #amount(Rs)

#Result
print "amount is",A,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
P=2000;    #principal(Rs)
R=10;       #rate(%)
T=3;       #time(yrs)

#Calculation
x=R/100;
y=(1+x)**T;
CI=P*(y-1);        #compound interest(Rs)

#Result
print "compound interest is",CI,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
R=10;       #rate(%)
T=2;       #time(yrs)
A=968;     #amount(Rs)

#Calculation
P=A/(1+(R/100))**T;     #principal(Rs)

#Result
print "principal is",P,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
P=64000;    #principal(Rs)
R=2.5;       #rate(%)
T=3;       #time(yrs)

#Calculation
x=R/100;
y=(1+x)**T;
CI=P*(y-1);        #compound interest(Rs)

#Result
print "compound interest is",CI,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
P=2000;    #principal(Rs)
R=8;       #rate(%)
T=9/12;       #time(yrs)
n1=4;
n2=2;
n3=1;

#Calculation
A1=P*(1+(R/(100*n1)))**(n1*T);   #amount(Rs)
CI1=A1-P;     #compound interest in 1st case(Rs)
A2=P*(1+(R/(100*n2)))**(n2*T);   #amount(Rs)
CI2=A2-P;     #compound interest in 2nd case(Rs)
A3=P*(1+(R/(100*n3)))**(n3*T);   #amount(Rs)
CI3=A3-P;     #compound interest in 3rd case(Rs)

#Result
print "compound interest in 1st case is",int(CI1),"Rs"
print "compound interest in 2nd case is",round(CI2,1),"Rs"
print "answer given in the book is wrong"
print "compound interest in 3rd case is",round(CI3),"Rs"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
R=4;       #rate(%)
T=2;       #time(yrs)
CI_SI=50;    #difference between CI and SI(Rs)

#Calculation
x=(R/100)**2;
P=CI_SI/x;     #sum(Rs)

#Result
print "sum is",P,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
R=10;       #rate(%)
T=2;       #time(yrs)
P=2000;     #sum(Rs)

#Calculation
x=(R/100)**2;
CI_SI=P*x;     #difference between CI and SI(Rs)

#Result
print "difference between CI and SI is",CI_SI,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
CI=282.15;      #compound interest(Rs)
SI=270;      #simple interest(Rs)
T=2;     #time(yrs)

#Calculation
R=(CI-SI)*100*T/SI;    #rate per annum(%)
P=100*SI/(R*T);     #sum(Rs)

#Result
print "rate per annum is",R,"%"
print "sum is",P,"Rs"
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
R=10;       #rate(%)
T=3;       #time(yrs)
CI_SI=31/2;   #difference between CI and SI(Rs)

#Calculation
x=R/100;
y=(x**3)+(T*(x**2));
P=CI_SI/y;     #sum(Rs)     

#Result
print "sum is",P,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
a=434;
b=272;

#Calculation
Nr=a-b;     #numerator of fraction
Dr=b;     #denominator of fraction
d=(Nr**2)-(4*Dr*Nr);       #discriminant
q=(-Nr-math.sqrt(abs(d)))/(2*Dr);    #solution
R=(-q-1)*100;         #rate(%)

#Result
print "rate percent is",round(R,3),"%"
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
R=5;      #rate(%)
A=50440;    #debt amount(Rs)
T=3;    #time(yrs)

#Calculation
x=100/(100+R);
y=x*(1+x+(x**2));
a=A/y;     #annual payment(Rs)

#Result
print "annual payment is",a,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
P=1;    #assume
A=1.44*P;    #amount(Rs)
T=2;    #time(yrs)
N=2;

#Calculation
R=(math.sqrt(A/P)-1)*100;      #rate(%)
T=(N-1)*100/R;     #time(yrs)

#Result
print "time is",T,"years"

#importing modules
import math
from __future__ import division

#Variable declaration
I2=238.50;    #interest in second year(Rs)
I1=225;       #interest in first year(Rs)
T=1;    #time(yrs)

#Calculation
I=I2-I1;     #interest on 1 year(Rs)
R=100*I/(I1*T);    #rate per annum(%)

#Result
print "rate per annum is",R,"%"

#importing modules
import math
from __future__ import division

#Variable declaration
CI_SI=20;     #difference for 2 years(Rs)
CISI=61;      #difference for 3 yrs(Rs)
T1=2;     #time(yrs)
T2=3;     #time(yrs)

#Calculation
R=((CISI/CI_SI)-T2)*100;    #rate(%)
P=CI_SI*(100/R)**2;     #sum(Rs)     

#Result
print "sum is",P,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
T=3;    #time(yrs)
P=1000;    #sum(Rs)
A=1331;    #amount(Rs)

#Calculation
x=(A/P)**(1/T);
R=100*(x-1);      #rate per annum(%)

#Result
print "rate per annum is",R,"%"

#importing modules
import math
from __future__ import division

#Variable declaration
P=64000;    #sum(Rs)
CI=4921;    #compound interest(Rs)
R=5;     #rate(%)
n=2;

#Calculation
A=P+CI;    #amount(Rs)
x=A/P;
y=(1+(R/(100*n)))**n;
T=math.log(x)/math.log(y);      #time(years)

#Result
print "time is",T,"years"

