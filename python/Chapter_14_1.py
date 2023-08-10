from math import pi

#variable declaration
cg=0.01;
gr=10**-8;

#calculation
l=cg/(gr*3600*24);
K_l_SCC=10;
a_sigma2=K_l_SCC**2/(1.21*pi);
s=[500,300,100];

#result
print('\nEstimated Life = %g days')%(l);
print('\n\n\n---------------------------------\nStress, MPa\tCrack Length, mm\n---------------------------------\n');
for i in range (0,3):
    print('\t%g\t\t%g\n')%(s[i],a_sigma2*1000/s[i]**2);
print('---------------------------------');

