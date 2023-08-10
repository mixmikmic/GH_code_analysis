#importing modules
import math
from __future__ import division

#Variable declaration
b=15;     #base(cm)
h=20;     #altitude(cm)

#Calculation
A=b*h/2;    #area of triangle(cm**2)

#Result
print "area of triangle is",A,"cm**2"

#importing modules
import math
from __future__ import division

#Variable declaration
A=205;    #area of triangle(cm**2)
s=41;     #side(cm)

#Calculation
p=2*A/s;    #perpendicular length(cm)

#Result
print "perpendicular length is",p,"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
a=5;
b=6;
c=7;       #sides of triangle(cm)

#Calculation
s=(a+b+c)/2;     #semi perimeter(cm)
A=math.sqrt(s*(s-a)*(s-b)*(s-c));    #base area(cm**2)

#Result
print "area is",int(A/math.sqrt(6)),"*math.sqrt(6) cm**2"

#importing modules
import math
from __future__ import division

#Variable declaration
a=4;     #side of triangle(cm)

#Calculation
A=math.sqrt(3)*a**2/4;    #area(cm**2)
P=3*a;      #perimeter(cm)

#Result
print "area is",round(A,3),"cm**2"
print "perimeter is",P,"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
b=10;    #base(cm)
a=13;    #side(cm)

#Calculation
A=b*math.sqrt((4*a**2)-(b**2))/4;    #area(cm**2)

#Result
print "area is",A,"cm**2"

#importing modules
import math
from __future__ import division

#Variable declaration
P=100;    #perimeter(cm)
b=36;     #base(cm)

#Calculation
a=(P-b)/2;       #length of each side(cm)

#Result
print "length of each side is",a,"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
l=20;      #length of one leg(cm)

#Calculation
A=l**2/2;     #area of triangle(cm**2)
P=l*(2+math.sqrt(2));     #perimeter of triangle(cm)

#Result
print "area of triangle is",A,"cm**2"
print "perimeter of triangle is",round(P,1),"cm"
print "answer given in the book is wrong"

#importing modules
import math
from __future__ import division

#Variable declaration
l1=12;     #length of one leg(cm)
l2=5;      #length of another leg(cm)

#Calculation
h=math.sqrt(l1**2+l2**2);     #length of hypotenuse(cm)
A=l1*l2/2;      #area of triangle(cm**2)

#Result
print "length of hypotenuse is",h,"cm"
print "area of triangle is",A,"cm**2"

#importing modules
import math
from __future__ import division

#Variable declaration
l=36;     #length(cm)
b=20;     #breadth(cm)

#Calculation
A=l*b;     #area of rectangle(cm**2)
P=2*(l+b);    #perimeter of rectangle(cm)

#Result
print "area of rectangle is",A,"cm**2"
print "perimeter of rectangle is",P,"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
l=12;      #length of rectangle(cm)
d=13;      #diagonal of rectangle(cm)

#Calculation
A=l*math.sqrt(d**2-l**2);     #area of rectangle(cm**2)
P=2*(l+math.sqrt(d**2-l**2));    #perimeter of rectangle(cm)

#Result
print "area of rectangle is",A,"cm**2"
print "perimeter of rectangle is",P,"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
l=20;     #length(cm)
b=15;     #breadth(cm)

#Calculation
d=math.sqrt(l**2+b**2);      #diagonal of rectangle(cm)

#Result
print "diagonal of rectangle is",d,"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
P=14;      #perimeter of rectangle(cm)
d=5;       #diagonal(cm)

#Calculation
A=((P**2/4)-(d**2))/2;     #area of rectangle(cm**2)

#Result
print "area of rectangle is",A,"cm**2"

#importing modules
import math
from __future__ import division

#Variable declaration
P=68;      #perimeter of rectangle(cm)
A=240;     #area of rectangle(cm**2)

#Calculation
l=A/10;     #length of rectangle(cm)
y=P/2;
b=y-l;      #breadth of rectangle(cm)

#Result
print "length of rectangle is",l,"cm"
print "breadth of rectangle is",b,"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
a=10;     #side of square(cm)

#Calculation
A=a**2;     #area(cm**2)
P=4*a;      #perimeter(cm)
d=a*math.sqrt(2);     #diagonal(cm)

#Result
print "area is",A,"cm**2"
print "perimeter is",P,"cm"
print "diagonal is",round(d,2),"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
A=900;     #area(m**2)

#Calculation
d=math.sqrt(2*A);     #diagonal(m)
P=math.sqrt(16*A);    #perimeter(m) 

#Result
print "diagonal is",A,"m"
print "perimeter is",P,"m"

#importing modules
import math
from __future__ import division

#Variable declaration
d=20;     #diagonal(cm)

#Calculation
A=d**2/2;     #area(cm**2)
P=math.sqrt(16*A);    #perimeter(cm)

#Result
print "area is",A,"cm**2"
print "perimeter is",round(P,3),"cm"

#importing modules
import math
from __future__ import division

#Variable declaration
d1=40;      #first diagonal(m)
d2=30;      #second diagonal(m)

#Calculation
A=d1*d2/2;     #area(m**2)
P2=4*(d1**2+d2**2);    
p=math.sqrt(P2);    #perimeter(m)

#Result
print "area is",A,"m**2"
print "perimeter is",p,"m"

#importing modules
import math
from __future__ import division

#Variable declaration
a=13;      #side of rhombus(cm)
h=20;      #height of rhombus(cm)

#Calculation
A=a*h;    #area of rhombus(cm**2)

#Result
print "area of rhombus is",A,"cm**2"

#importing modules
import math
from __future__ import division

#Variable declaration
a=25;      #side of rhombus(m)
h=40;      #height of rhombus(m)

#Calculation
A=d*math.sqrt((a**2)-((h/2)**2));      ##area of rhombus(m**2)
P=4*a;      #perimeter of rhombus(m)

#Result
print "area of rhombus is",A,"m**2"
print "perimeter of rhombus is",p,"m"

#importing modules
import math
from __future__ import division

#Variable declaration
d=12;      #diagonal(cm)
p1=13;     #length of offset 1(cm)
p2=7;      #length of offset 2(cm)

#Calculation
A=d*(p1+p2)/2;    #area of quadrilateral(m**2)

#Result
print "area of quadrilateral is",A,"m**2"

#importing modules
import math
from __future__ import division

#Variable declaration
d=30;      #length of diagonal(m)
l=20;      #length of perpendicular(m)

#Calculation
A=d*l;    #area of parallelogram(m**2)

#Result
print "area of parallelogram is",A,"m**2"

#importing modules
import math
from __future__ import division

#Variable declaration
a=12;     #length of side 1(m)
b=14;     #length of side 2(m)
d=22;     #length of diagonal(m)

#Calculation
S=(a+b+d)/2;    #semi perimeter(m)
A=2*math.sqrt(S*(S-a)*(S-b)*(S-d));     #area of parallelogram(m**2)

#Result
print "area of parallelogram is",round(A,3),"m**2"

#importing modules
import math
from __future__ import division

#Variable declaration
a=13;     #length of side 1(m)
b=11;     #length of side 2(m)
d1=16;     #length of diagonal 1(m)

#Calculation
d22=(2*(a**2+b**2))-(d1**2);    
d2=math.sqrt(d22);      #length of second diagonal(m)

#Result
print "length of second diagonal is",d2,"m"

#importing modules
import math
from __future__ import division

#Variable declaration
a=20;     #length of parallel side 1(m)
b=25;     #length of parallel side 2(m) 
h=12;     #distance(m)

#Calculation
A=(a+b)*h/2;   #area of trapezium(m**2)

#Result
print "area of trapezium is",A,"m**2"

#importing modules
import math
from __future__ import division

#Variable declaration
a=120;     #length of parallel side 1(m)
b=75;     #length of parallel side 2(m) 
c=105;     #length of non parallel side 1(m)
d=72;     #length of non parallel side 2(m) 

#Calculation
K=a-b;    #difference of parallel sides(m)
S=(K+c+d)/2;    #semi perimeter(m)
x=S*(S-K)*(S-c)*(S-d);    
A=(a+b)*math.sqrt(x)/K;    #area of trapezium(m**2)

#Result
print "area of trapezium is",round(A,2),"m**2"
print "answer varies due to rounding off errors"

#importing modules
import math
from __future__ import division

#Variable declaration
r=14;     #radius(m)

#Calculation
C=2*math.pi*r;     #circumference of circle(m)
A=math.pi*r**2;    #area of circle(m**2)

#Result
print "area of circle is",round(A),"m**2"
print "circumference of circle is",round(C),"m"

#importing modules
import math
from __future__ import division

#Variable declaration
C=44;     #circumference of circle(m)

#Calculation
A=C**2/(4*math.pi);    #area of circle(m**2)

#Result
print "area of circle is",round(A),"m**2"

