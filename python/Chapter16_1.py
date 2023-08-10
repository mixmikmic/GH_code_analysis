#importing modules
import math
from __future__ import division

#Variable declaration
P=176;    #principal(Rs)
R=6;       #rate(%)
T=20/12;       #time(yrs)

#Calculation
PW=P/(1+(R*T/100));    #present worth(Rs)
TD=P-PW;         #true discount(Rs)

#Result
print "present worth is",PW,"Rs"
print "true discount is",TD,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
P=2420;    #principal(Rs)
R=10;       #rate(%)
T=2;       #time(yrs)

#Calculation
PW=P/((1+(R/100))**T);    #present worth(Rs)
TD=P-PW;         #true discount(Rs)

#Result
print "present worth is",PW,"Rs"
print "true discount is",TD,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
R=12;       #rate(%)
T=8/12;       #time(yrs)
TD=240;         #true discount(Rs)

#Calculation
PW=100*TD/(R*T);    #present worth(Rs)
P=PW+TD;     #amount of the bill(Rs)

#Result
print "present worth is",PW,"Rs"
print "amount of the bill is",P,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
P=960;    #principal(Rs)
R=5;       #rate(%)
T=4;       #time(yrs)

#Calculation
SI_TD=(P*(R*T)**2)/(100*(100+(R*T)));       #difference between simple interest and true discount(Rs)

#Result
print "difference between simple interest and true discount is",SI_TD,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
R=4;       #rate(%)
T=6/12;       #time(yrs)
SI_TD=15;     #difference between simple interest and true discount(Rs)

#Calculation
P=SI_TD*100*(100+(R*T))/((R*T)**2);       #principal(Rs)

#Result
print "principal is",P,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
P=2040;    #principal(Rs)
R=6;       #rate(%)
TD=40;     #true discount(Rs)

#Calculation
T=100*TD/(R*(P-TD));     #time(yrs)

#Result
print "time is",T*12,"months"

#importing modules
import math
from __future__ import division

#Variable declaration
P=1860;    #principal(Rs)
T=3;       #time(yrs)
TD=60;     #true discount(Rs)

#Calculation
R=100*TD/(T*(P-TD));    #rate(%)  

#Result
print "rate is",round(R,2),"%"
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
T=5/12;       #time(yrs)
TD=75;     #true discount(Rs)
R=12;       #rate(%)

#Calculation
P=TD*(1+(100/(R*T)));    #amount of the bill(Rs)

#Result
print "amount of the bill is",P,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
TD=200;     #true discount(Rs)
SI=300;     #simple interest(Rs)
T=3;       #time(yrs)

#Calculation
P=SI*TD/(SI-TD);      #sum(Rs)
R=100*SI/(P*T);       #rate per annum(%)

#Result
print "sum is",P,"Rs"
print "rate per annum is",round(R,2),"%"

#importing modules
import math
from __future__ import division

#Variable declaration
TD1=60;     #true discount(Rs)
T1=1;     #assume
T2=2*T1;
R1=R2=1;     #assume
P=240;       #sum(Rs)

#Calculation
x=(P-TD1)/(T2*TD1);
TD2=P/(x+1);      #true discount(Rs)

#Result
print "true discount is",TD2,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
PW1=1500;    #present worth(Rs)
PW2=1200;    #present worth(Rs)
T1=6/12;       #time(yrs)
T2=2;     #time(yrs)

#Calculation
x=PW1/PW2;
R=100*(x-1)/(T2-(x*T1));     #rate percent(%)
P=PW1*(1+(R*T1/100));       #sum of the bill(Rs)

#Result
print "rate percent is",round(R,2),"%"
print "sum of the bill is",round(P,2),"Rs"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
P1=600;       #sum(Rs)
P2=720;       #sum(Rs)
T=5;       #time(yrs)

#Calculation
R=((P2*T*100/(P1*T))-100)/T;      #rate percent(%)

#Result
print "rate percent is",R,"%"

#importing modules
import math
from __future__ import division

#Variable declaration
P=8480;       #sum(Rs)
n=4;    #number of installments
r=16;    #rate percent(%)
y=4;

#Calculation
x=r*n*(n-1)/(100*y*2);
a=P/(x+n);    #each installment should be(Rs)

#Result
print "each installment should be",a,"Rs"

