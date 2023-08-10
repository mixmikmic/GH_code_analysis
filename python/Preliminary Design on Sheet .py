Bm=0.35;
Bsat=0.5;

Bm=0.35;#in units of tesla
Bmgau=0.35*1e4; #in units of gauss
Bsat=0.5; #in units of tesla
Bsatgau=0.5*1e4;  #in units of gauss
V1=417; #applied peak voltage to primary winding
V1rms=417; #applied rms voltage to primary winding
V2=12500; #obtained peak voltage at secondary winding
V2rms=12500; #applied rms voltage to primary winding
Pout=30000; #rated output power
fsw=100000; #switching frequency of transformer
fswmax=500000; #maksimum switching frequency of transformer
pi=3.141592;
a=V1/V2 #turns ratio of transformer
N1Ac=(V1rms*1e8)/(Bsatgau*fsw*4);
print (N1Ac)#minimum required N1.Ac product for designed transformer
print("The minimum required N1.AC product is",N1Ac)
#Ac will be determined by chosen core,therefore N1 will be calculated from this product

C=5.07e-3; #current capacity in cm2/ampere
e=0.95; #efficiency of the transformer
K=0.3; #winding factor
WaAc=(Pout*C*1e8)/(4*e*Bsatgau*fsw*K)
print (WaAc)
print("The required power handling capacity is",WaAc,"in units of cm^4")

from IPython.display import Image 
Image(filename='core.jpg')

from IPython.display import Image 
Image(filename='power capacity.jpg')

#...................49928, P material, ferrite core was chosen..........................#
WaAc=90.6; #cm^4
AL=6.773e-6; #H/turn for P type material
le=274; #in mm,effective length
Ac=738; #in mm2,effective cross sectional area
Accm2=Ac/1e2; #in cm2,effective cross sectional area
Wa=WaAc/Accm2;
Ve=202000; #mm3
Weight=980; #grams per set of the core
#lengths of the parts of the cores are at below.
A=100.3; #in units of mm
B=59.4;#in units of mm
C=27.5;#in units of mm
D=46.85;#in units of mm
E=72;#in units of mm
F=27.5;#in units of mm
L=13.75;#in units of mm
M=22.65;#in units of mm

N1=(N1Ac)/(Accm2); # The N1Ac changed cm2 to mm2
print (N1)
N1opt=round (N1);
print (N1opt)
print("The optimum number of turns for primary winding is",N1opt)

N2=N1/a;
print (N2)
N2opt=round (N2);
print (N2opt)
print("The optimum number of turns for primary winding is",N2opt)

core_window_area=2*D*M*2;
core_window_areacm2=core_window_area/100;
print("The obtained core total windows area is",core_window_areacm2,"in units of cm^2")

I2rms=Pout/V2rms;
#in units of Amper
print("The secondary rms current is",I2rms,"in units of amps.")

I1rms=I2rms/a;
print("The primary rms current is",I1rms,"in units of amps")

j=3; #current density in units of A/mm2
CAc_I2=I2rms/j;
print (CAc_I2);#in units of mm2
radius_square_Cac_I2=CAc_I2/pi;
radius_CAc_I2=radius_square_Cac_I2**(1/2);
diameter_CAc_I2=radius_CAc_I2*2;
print (diameter_CAc_I2); #in units of mm

CAc_I1=I1rms/j;
print (CAc_I1); #in units of mm2

skdp=0.066/(fswmax**(1/2)); #skin depth in units of m. By thinking of pure "COPPER" was chosen.
skdpmm=skdp*1e3; #skin depth in units of mm
print (skdpmm);

Area_AWG3=26.7;#in units of mm2 for primary current
Area_AWG18=0.823;#in units of mm2 for primary current
radius_square_AWG18=Area_AWG18/pi;
radius_AWG18=radius_square_AWG18**(1/2);
diameter_AWG18=radius_AWG18*2;
print (diameter_AWG18);

Area_AWG33=0.0254;#in units of mm2
radius_square_AWG33=Area_AWG33/pi;
radius_AWG33=radius_square_AWG33**(1/2);
diameter_AWG33=radius_AWG33*2;
print (diameter_AWG33);

required_number_parallel_secondary=CAc_I2/Area_AWG33;
print (required_number_parallel_secondary); #required number of parallel windings for secondary windings

kw=0.3; #winding factor for E geometry cores
rqrd_area_scndry=Area_AWG33*required_number_parallel_secondary*N2opt/kw; #AWG in units of mm2
rqrd_area_scndrycm2=rqrd_area_scndry/100;
print("The required window area for secondary windings",rqrd_area_scndrycm2,"in units of cm2");

required_number_parallel_primary=CAc_I1/Area_AWG33;
print("The required number of parallel windings for secondary windings",required_number_parallel_primary);
rqrd_area_primary=Area_AWG33*required_number_parallel_primary*N1opt/kw; #AWG in units of mm2
rqrd_area_primarycm2=rqrd_area_primary/100;
print("The required window area for primary windings",rqrd_area_primary,"in units of mm2");
print("The required window area for primary windings",rqrd_area_primarycm2,"in units of cm2");

print("The required window area for primary windings",rqrd_area_primarycm2,"in units of cm2");
print("The required window area for secondary windings",rqrd_area_scndrycm2,"in units of cm2");
print("The obtained core total windows area is",core_window_areacm2,"in units of cm^2");

required_number_parallel_secondaryopt=round(required_number_parallel_secondary)
required_number_parallel_primaryopt=round(required_number_parallel_primary)
print("The primary number of turns",N1opt," and the primary number of parallel windings",required_number_parallel_primaryopt )
print("The secondary number of turns",N2opt," and the secondary number of parallel windings",required_number_parallel_secondaryopt )

winding_primary=N1opt*required_number_parallel_primaryopt;
print (winding_primary);
N1st=2*D/(diameter_AWG33);
print (N1st);
total_winding=(winding_primary)/N1st;
print (total_winding);
print("Number of ",total_winding,"fullturn is needed to totally wind primary")
length_primary=(2*pi*(F/2))*N1st+(2*pi*((F/2)+diameter_AWG33)*N1st)+(2*pi*((F/2)+diameter_AWG33*2)*N1st);
length_primary_con=(2*pi*((F/2)+diameter_AWG33*3)*N1st)+(2*pi*((F/2)+diameter_AWG33*4)*N1st);
length_primary_continue=((2*pi*((F/2)+diameter_AWG33*5)*N1st)*(1/0.4352));
total_length_primary=length_primary+length_primary_con+length_primary_continue;
print("Total length for primary winding",total_length_primary,"in units of mm");
total_length_primary

print (skdpmm); #in units of mm
#Starting with primary side.
#Area_AWG3=26.7;#in units of mm2 for primary current
#Area_AWG18=0.823;#in units of mm2 for primary current
radius_square_AWG3=Area_AWG3/pi;
radius_AWG3=radius_square_AWG3**(1/2);
diameter_AWG3=radius_AWG3*2;
print (diameter_AWG3);
wastedarea_primary=pi*((radius_AWG3-skdpmm)**2)
totalarea_primary=pi*(radius_AWG3**2);
effectivearea_primary=totalarea_primary-wastedarea_primary;

required_number_parallel_primary_new=CAc_I1/effectivearea_primary;
required_number_parallel_primaryopt_new=round(required_number_parallel_primary_new);
print ("The required number of parallel windings for primary windings", required_number_parallel_primaryopt_new); 

winding_primary_new=N1opt*required_number_parallel_primaryopt_new;
print (winding_primary_new);
N1st_new=2*D/(diameter_AWG3);
print (N1st_new);
total_winding_new=(winding_primary_new)/N1st_new;
print (total_winding_new);
print("Number of ",total_winding_new,"fullturn is needed to totally wind primary")
length_primary_new=(2*pi*(F/2))*N1st_new+(2*pi*((F/2)+diameter_AWG3*1)*N1st_new);
length_primary_continue_new=((2*pi*((F/2)+diameter_AWG3*2)*N1st_new)*(0.613));
total_length_primary_new=length_primary_new+length_primary_continue_new;
print("Total length for primary winding new is",total_length_primary_new,"in units of mm");
total_length_primary_new_m=total_length_primary_new/100;#in units of meter
print("Total length for primary winding new is",total_length_primary_new_m,"in units of m");

print (skdpmm); #in units of mm
#Going on with secondary side.
#Area_AWG3=26.7;#in units of mm2 for primary current
#Area_AWG18=0.823;#in units of mm2 for primary current
radius_square_AWG18=Area_AWG18/pi;
radius_AWG18=radius_square_AWG18**(1/2);
diameter_AWG18=radius_AWG18*2;
print (diameter_AWG18);
wastedarea_secondary=pi*((radius_AWG18-skdpmm)**2)
totalarea_secondary=pi*(radius_AWG18**2);
effectivearea_secondary=totalarea_secondary-wastedarea_secondary;

required_number_parallel_secondary_new=CAc_I2/effectivearea_secondary;
required_number_parallel_secondaryopt_new=round(required_number_parallel_secondary_new);
print ("The required number of parallel windings for secondary windings", required_number_parallel_secondaryopt_new); 

winding_secondary_new=N2opt*required_number_parallel_secondaryopt_new;
print("Total winding turn for secondary winding new is",winding_secondary_new)
N2st_new=2*D/(diameter_AWG18);
print (N2st_new);
total_winding_secondary_new=(winding_secondary_new)/N2st_new;
print (total_winding_secondary_new);
print("Number of ",total_winding_secondary_new,"fullturn is needed to totally wind secondary")
length_secondary_new=(2*pi*(((F/2)+diameter_AWG18*3)+diameter_AWG3*1)*N2st_new);
length_secondary_new_cont=(2*pi*(((F/2)+diameter_AWG18*3)+diameter_AWG3*2)*N2st_new);
length_secondary_new_continue=(2*pi*(((F/2)+diameter_AWG18*3)+diameter_AWG3*3)*N2st_new)*(0.7858);
total_length_secondary_new=length_secondary_new+length_secondary_new_cont+length_secondary_new_continue;
print("Total length for secondary winding new is",total_length_secondary_new,"in units of mm");
total_length_secondary_new_m=total_length_secondary_new/100;#in units of meter
print("Total length for secondary winding new is",total_length_secondary_new_m,"in units of m");

AWG3_res=0.6465e-3; #ohm/meter
AWG18_res=20.95e-3; #ohm/meter
Rpri_1x_length=total_length_primary_new_m/required_number_parallel_primaryopt_new;
Rpri_1x=Rpri_1x_length*AWG3_res
Rpri_equi=Rpri_1x/required_number_parallel_primaryopt_new;
print("Total equivalent resistance of primary winding new is",Rpri_equi,"in units of ohm");
Rsec_1x_length=total_length_secondary_new_m/required_number_parallel_secondaryopt_new;
Rsec_1x=Rsec_1x_length*AWG18_res
Rsec_equi=Rsec_1x/required_number_parallel_secondaryopt_new;
print("Total equivalent resistance of secondary winding new is",Rsec_equi,"in units of ohm");

Ppri_copperloss=(I1rms**2)*Rpri_equi;
Psec_copperloss=(I2rms**2)*Rsec_equi;
print("Total equivalent copper loss of primary winding new is",Ppri_copperloss,"in units of Watts");
print("Total equivalent copper loss of secondary winding new is",Psec_copperloss,"in units of Watts");

from IPython.display import Image 
Image(filename='49928.jpg')

from math import log
set=2; #for double EE configuration
Pcore_given=34; #W/set given from datasheet @100kHz,100mT,100C
Pcore_given_condition=Pcore_given*set; #W @100mT, it should be converted to Bmax(T)
Pcoreloss=Pcore_given_condition*log(500,10)/log(100,10);
print("Total equivalent coreloss of the material is",Pcoreloss,"in units of w");

totallosses=Pcoreloss+Ppri_copperloss+Psec_copperloss;
print("Total equivalent loss of the transformer is",totallosses,"in units of w");
efficiency=Pout/(Pout+totallosses);
print("Total efficiency of the transformer is",efficiency);

weightofcore=980;#grams/set
totalweightofcore=weightofcore*set;
totalweightofcorekg=totalweightofcore/1000;
print("Total weight of core is",totalweightofcorekg,"in units of kg");
AWG3_weightlb_1000ft=159.3;#lb/1000ft
AWG18_weightlb_1000ft=4.917;#lb/1000ft
#to change lb/1000ft to kg/km  1lb/1000ft=1.48816kg/km
constant_lb2kg_meter=0.00148816; #kg/m conversion
AWG3_weightkg_meter=AWG3_weightlb_1000ft*constant_lb2kg_meter;
AWG18_weightkg_meter=AWG18_weightlb_1000ft*constant_lb2kg_meter;
weight_secondary_winding=total_length_secondary_new_m*AWG18_weightkg_meter; #in kg
weight_primary_winding=total_length_primary_new_m*AWG3_weightkg_meter; #in kg
print("Total weight of primary winding is",weight_primary_winding,"in units of kg");
print("Total weight of secondary is",weight_secondary_winding,"in units of kg");

Lpri=(N1opt**2)*AL; #henry
Lpri_uhenry=Lpri*1e6;
print("Primary inductance of the transformer is",Lpri_uhenry,"in units of uH");
Lpri_H=Lpri_uhenry*1e-6;

Lsec=Lpri_uhenry*(N2opt**2)/(N1opt**2)
Lsec_mh=Lsec/1e3;
print("Secondary inductance of the transformer is",Lsec_mh,"in units of mH");
Lsec_H=Lsec_mh*1e-3;

Mutual_inductance=((Lsec_H*Lpri_H)**(1/2));
print (Mutual_inductance)

k=1/30; #practical value
lmag_pri=0.98*Lpri_H;
lmag_priuH=lmag_pri*1e6
lleak_pri=Lpri_H*(1-k);
lleak_priuH=lleak_pri*1e6;
#referring to primary
lleak_sec_referred=Lsec_H*k;
lleak_sec_referredmH=lleak_sec_referred*1e3
rsec_reffered=Rsec_equi*(a**2);
print("Magnetizing inductance of the primary transformer is",lmag_priuH,"in units of uH");
print("Primary leakage inductance of the primary transformer is",lleak_priuH,"in units of uH");
print("Primary winding resistance is",Rpri_equi,"in units of ohm");
print("Secondary leakage inductance referred to primary is",lleak_sec_referredmH,"in units of mH");
print("Secondary winding resistance referred to primary is",rsec_reffered,"in units of ohm");





