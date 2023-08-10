#importing modules
import math
from __future__ import division

#Variable declaration
a1=5000;     #amount(Rs)
mv1=95;    #market value
a2=3000;     #amount(Rs)
a3=2500;     #amount(Rs)
mv2=100+10;  #market value
a4=1000;     #amount(Rs)
d1=8;    #discount(%)
a5=2200;     #amount(Rs)
b1=1/11;     #brokerage(%)
a6=2000;     #amount(Rs)
mv3=100+5;   #market value
b2=1/2;     #brokerage(%)
d2=6;       #discount
a7=1000;    #amount(Rs)
b3=1/2;     #brokerage(%)

#Calculation
cp1=a1*mv1/100;     #cost of purchase in 1st case(Rs)
cp2=a2;      #cost of purchase in 2nd case(Rs)
cp3=a3*mv2/100;   #cost of purchase in 3rd case(Rs) 
cp4=a4*(100-d1)/100;    #cost of purchase in 4th case(Rs) 
cp5=a5*(100+b1)/100;       #cost of purchase in 5th case(Rs) 
cp6=a6*(mv3+b2)/100;   #cost of purchase in 6th case(Rs) 
mv4=100-d2;     #market value
cp7=a7*(mv4+b3)/100;   #cost of purchase in 7th case(Rs)  

#Result
print "cost of purchase in 1st case is",cp1,"Rs"
print "cost of purchase in 2nd case is",cp2,"Rs"
print "cost of purchase in 3rd case is",cp3,"Rs"
print "cost of purchase in 4th case is",cp4,"Rs"
print "cost of purchase in 5th case is",cp5,"Rs"
print "cost of purchase in 6th case is",cp6,"Rs"
print "cost of purchase in 7th case is",cp7,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
a=2000;     #amount(Rs)
mv=100+6;   #market value
b=1/2;     #brokerage(%)

#Calculation
sr=a*(mv-b)/100;    #sale realisation(Rs) 

#Result
print "sale realisation is",sr,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
mv=100+(9/4);   #market value
b=1/2;     #brokerage(%)
sr=1221;     #sale realisation(Rs) 
 
#Calculation
a=sr*100/(mv-b);    #amount of stock(Rs) 

#Result
print "amount of stock is",a,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
b=1/2;     #brokerage(%)
d=8;    #discount(%)
pc=1850;   #purchase cost(Rs)

#Calculation
a=pc*100/(100-d+b);    #amount of stock(Rs) 

#Result
print "amount of stock is",a,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
a=2800;     #amount of stock(Rs) 
r=4/100;    #rate of stock(%)

#Calculation
I=a*r    #annual income(Rs)

#Result
print "annual income is",I,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
i=2800;     #investment(Rs)
r=4;    #rate of stock(%)
mvb=112;    

#Calculation
I=i*r/mvb;    #annual income(Rs)

#Result
print "annual income is",I,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
i=100;     #investment(Rs)
r=7;    #rate of stock(%)
b=1/(4*100);     #brokerage(%)
mv=1-(5/100);    #market value

#Calculation
R=i*r/(mv+b);   #rate percent obtained(%)

#Result
print "rate percent obtained is",round(R/100,2),"%"

#importing modules
import math
from __future__ import division

#Variable declaration
i=1220;     #investment(Rs)
r=6;    #rate of stock(%)
b=1/4;     #brokerage(%)
I=244;    #annual income(Rs)

#Calculation
mv=(i*r/I)-b;     #market value

#Result
print "market value is",mv

#importing modules
import math
from __future__ import division

#Variable declaration
r=20/3;    #rate of stock(%)
mvb=110;
b=1/4;     #brokerage(%)
i=300;    #annual income(Rs)

#Calculation
I=i*mvb/r;   #investment(Rs)

#Result
print "investment is",I,"Rs"
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
fv=20;    #face value(Rs)
mv=74;    #market value(Rs)
n=250;    #number of shares

#Calculation
a=mv*n;    #amount paid by buyer(Rs)
cp=fv*n;   #purchase cose(Rs)
g=a-cp;    #gain by share holder(Rs)

#Result
print "gain by share holder is",g,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
pc=4220;    #purchase cost(Rs)
mv=105;     #market value
b=1/2;     #brokerage(%)

#Calculation
sr=pc*(mv-b)/(mv+b);    #sale realisation(Rs) 

#Result
print "sale realisation is",sr,"Rs"

#importing modules
import math
from __future__ import division

#Variable declaration
fv=10;    #face value(Rs)
d=3/8;    #discount
b=1/8;     #brokerage(%)
n=80;    #number of shares

#Calculation
c1=fv-d+b;    #cost of 1 share(Rs)
C=n*c1;      #cost of 80 shares(Rs)

#Result
print "cost of 80 shares is",C,"Rs"

#importing modules
import math
from __future__ import division
from fractions import gcd

#Variable declaration
mvb=120;
a=4500;     #amount(Rs)
r=5;    #rate of stock(%)
i=75;   #income(Rs)
x1=99;
x2=132; 
r1=3;    #rate(%) 
r2=8;    #rate(%)

#Calculation
sr=mvb*a/100;     #sale realisation(Rs) 
Is=a*r/100;     #income before selling(Rs)
Ias=Is+i;     #income after sale(Rs)


l=x1*x2/gcd(x1,x2);     #lcm of x1 and x2
X=l*Ias;
f1=l/x1;
f2=l/x2;
c1=r2*f1;    #c1=r2*f1
c2=r1*f1;
c=l*r2*sr/x2;
x=(c-X)/(c1-c2);    #amount invested in 3% stock(Rs)
y=sr-x;     #amount invested in 8% stock(Rs)

#Result
print "amount invested in 3% stock is",x,"Rs"
print "amount invested in 8% stock is",y,"Rs"
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division
from fractions import gcd

#Variable declaration
i=2592;    #investment(Rs)
mvb=108;
fv=100;   #face value(Rs)
d=25/2;   #dividend(%)

#Calculation
I=i*d*fv/(mvb*100);    #income derived(Rs)

#Result
print "income derived is",I,"Rs"

#importing modules
import math
from __future__ import division
from fractions import gcd

#Variable declaration
r=17/4;    #rate(%)
fv=20;   #face value(Rs)
n=88;    #number of shares
p=5;    #premium
b=1/4;   #brokerage(%)

#Calculation
d=r*fv*n/100;     #dividend(Rs)
pc=fv+p+b;    #purchase cost(Rs)
R=r*fv/pc;    #rate of interest on investment(%)

#Result
print "rate of interest on investment is",round(R,2),"%"

#importing modules
import math
from __future__ import division
from fractions import gcd

#Variable declaration
i=4444;    #investment(Rs)
I=600;     #annual income(Rs)
fv=100;   #face value(Rs)
d=15;   #dividend(%)
b=1/100;   #brokerage(%)

#Calculation
M=i*d*fv/(I*100*(1+b));    #market value of each share(Rs)

#Result
print "market value of each share is",M,"Rs"

#importing modules
import math
from __future__ import division
from fractions import gcd

#Variable declaration
I=1500;     #annual income(Rs)
fv=100;   #face value(Rs)
d=15;   #dividend(%)
b=1/100;   #brokerage(%)
M=104;   #market value(Rs)

#Calculation
i=I*100*M*(1+b)/(d*fv);    #investment(Rs)

#Result
print "investment is",i,"Rs"

#importing modules
import math
from __future__ import division
from fractions import gcd

#Variable declaration
d1=15;     #debenture(%)
d2=14;     #debenture(%)
p=8;      #premium(%)
d=4;      #discount(%)

#Calculation
M1=100-d;    #market value(Rs)
M2=100+p;    #market value(Rs)
x=d1*M1;     #investment(Rs)
y=d2*M2;     #investment(Rs)
if(x<y):
    print "investment is better"
else:
    print "investment is not better"
    
#Result
print "investment in 1st case is",x,"Rs"
print "investment in 2nd case is",y,"Rs"
print "answer given in the book for 2nd case is wrong"

