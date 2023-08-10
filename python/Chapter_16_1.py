from math import sqrt
from math import exp
from math import log

#variable declaration
sigma=1000;
mu=0.25;
a=2;
b=6;
h=0.25;
x=0;
mu=0.25;

#calculation
p_max=2*sigma*exp(2*mu*(a-x)/h)/sqrt(3);
print('\nAt the centerline of the slab = %g psi\n')%(p_max);
print('\nPressure Distributon from the centerline:');
print('\n---------------------------------\n');
print('x\tp (ksi)\t\tt_i (ksi)\n');
print('---------------------------------\n');
while x<=2:
    p=2*sigma*exp(2*mu*(a-x)/h)/(1000*sqrt(3));             #in ksi
    t_i=mu*p;
    print('%g\t%g\t\t%g\n')%(x,p,t_i);
    x+=0.25;
print('---------------------------------\n');
k=sigma/sqrt(3);
x=0;
p_max1=2*sigma*((a-x)/h+1)/sqrt(3);
x1=a-h/(2*mu)*log(1/(2*mu));
p=2*sigma*(a/(2*h)+1)/sqrt(3);
P=2*p*a*b;
P=P*0.000453;                      #conversion to metric tons

#result
print('\nFor sticking friction:\np_max = %g ksi')%(p_max1/1000);
print('\n\nThe Forging load = %g tons')%(P);

