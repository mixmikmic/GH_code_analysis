
#variable declaration
sf=3;
per=1/1000;
T=[1100, 1500];
C=[30000, 4000];              
W=[0, 0];
#calculation
W[0]=C[0]/sf;
W[1]=C[1]/sf;

#result
print('\n-----------------------------------------------------------\n');
print('Temperature\tCreep Strength, psi\tWorking Stress, psi\n');
print('------------------------------------------------------------');
print('\n1100 F\t\t\t%i\t\t\t%i\n')%(C[0],W[0]);
print('\n1500 F\t\t\t%i\t\t\t%i\n')%(C[1],W[1]);



from math import log

#variable declaration
def C(f):
    return (f-32)*5/9;
R=1.987;
T2=1300;
T1=1500;

#calculation
T2=C(T2)+273.15;
T1=C(T1)+273.15;
e2=0.0001;
e1=0.4;
Q=R*log(e1/e2)/(1/T2-1/T1);

#result
print('\nActivation Energy = %g cal/mol')%(Q)
print('\n\n\nNote: Calculation Errors in book');


from math import log10

#variable declaration
t=10**5;
C1=20;
T1=1200;
T2=1600;

#calculation
P_1200=(T1+460)*(log10(t)+C1);
P_1600=(T2+460)*(log10(t)+C1);

#result
print('\nAt T = 1200 F, P = %g\nAt T = 1600 F, P = %g\nAnd from the master ploy of Astroploy, corresponding stress required are sigma = 78000 psi and sigma = 11000 psi')%(P_1200,P_1600);

