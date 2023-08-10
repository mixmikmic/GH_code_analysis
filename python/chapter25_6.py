#Variable declaration
A=100.0;              #Open-circuit voltage gain of differential amplifier
V1=3.25;              #Input voltage to terminal 1 in V
V2=3.15;              #Input voltage to terminal 2 in V

#Calculations
V0=A*(V1-V2);         #Output voltage in V

#Results
print("The output voltage of the differential amplifier = %dV"%V0);

from math import log10

#Variable declaration
A_DM=2000.0;              #Differential mode voltage gain
A_CM=0.2;                 #Common mode voltage gain

#Calculations
CMRR=A_DM/A_CM;           #Common mode rejection ratio
CMRR_dB=20*log10(CMRR);     #Common mode rejection ratio in dB


#Results
print("The common mode rejection ratio = %d."%CMRR);
print("The common mode rejection ratio in decibels= %ddB."%CMRR_dB);

from math import log10

#Variable declaration
VD_in=10.0;               #Differential mode input in mV
VD_out=1.0;               #Output for differential mode input in V
VC_in=10.0;                 #Common mode input in mV
VC_out=5.0;                 #Output for common mode input in mV\

#Calculations
A_DM=(VD_out*1000)/VD_in;       #Differntial mode voltage gain
A_CM=VC_out/VC_in;       #Common mode voltage gain
CMRR=A_DM/A_CM;           #Common mode rejection ratio
CMRR_dB=20*log10(CMRR);     #Common mode rejection ratio in dB

#Results
print("The common mode rejection ratio in decibels= %ddB."%CMRR_dB);

#Variable declaration
A_DM=150.0;           #Differential mode voltage gain
CMRR_dB=90.0;         #Common mode rejection ratio
V1=100.0;             #Input voltage for terminal 1 in mV
V2=50.0;              #Input voltage for terminal 2 in mV
V_noise=1.0;          #Voltage of noise signal in mV

#Calculation

#Case(i)
V_out=A_DM*(V1-V2)/1000.0;            #Output voltage for differntial mode input, in V

#Since CMRR_dB=20*log10(differential mode gain/common mode gain),
A_CM=A_DM/pow(10,(CMRR_dB/20));         #Common mode gain
V_OUT_noise=A_CM*(V_noise/1000);        #Noise on output in V


#Results
print("Output voltage =%.1fV"%V_out);
print("Noise on output = %.1fx10^-6V"%(V_OUT_noise*pow(10,6)));

from math import log10

#Variable declaration
A_DM=2500.0;                    #Differential mode voltage gain
CMRR=30000.0;                   #Common mode rejection ratio
Input_signal=500.0;             #Single ended input r.m.s signal in microvolts
Interference=1.0;               #Interference signal, in V

#Calculations

#(i)
A_CM=A_DM/CMRR;                 #Common mode gain

#(ii)
CMRR_dB=20*log10(CMRR);         #Common mode rejection ratio in decibels

#(iii)
V_out=A_DM*(Input_signal/pow(10,6)-0);        #r.m.s output signal  in V

#(iv)
Interference_out=A_CM*Interference;          #r.m.s interference output in V
Interference_out=Interference_out*1000;      #r.m.s interference output in mV


#Results
print("Common mode gain =%.3f"%A_CM);
print("Common mode rejection ratio in decibels=%.1fdB"%CMRR_dB);
print("r.m.s output signal =%.2fV"%V_out);
print("r.m.s interfernce output voltage = %dmV"%Interference_out);

#Variable declaration
VCC=12;                 #Collector supply voltage, V
VEE=12;                 #Emitter supply voltage, V
RB=10;                  #Base resistor, kΩ
RC2=10;                 #Collector resistor, kΩ
RE=25;                  #Emitter resistor, kΩ
VBE=0.7;               #Base-emitter voltage, V
beta=100;               #Base amplification factor

#Calculation
VE=-VBE;                    #Emitter voltage, V (Ignoring the base current)
IE=(VEE-VBE)/RE;            #Tail current, mA
IE1=IE/2;                   #Emitter current of 1st transistor, mA
IE2=IE1;                    #Emitter current of 2nd transistor, mA
IC1=IE1;                    #Collector current(= emitter current) of 1st transistor, mA
IC2=IC1;                    #Collector current of 2nd transistor, mA
IB1=(IC1/beta)*1000;        #Base current of 1st transistor, μA
IB2=IB1;                    #Base current of 2nd transistor, μA
VC1=VCC;                    #Collector voltage of 1st transistor, V
VC2=VCC-IC2*RC2;            #Collector voltage of 2nd transistor, V

#Result
print("VE=%.1fV"%VE);
print("IE=%.3fmA"%IE);
print("IE1=%.3fmA"%IE1);
print("IE2=%.3fmA"%IE2);
print("IC1=%.3fmA"%IC1);
print("IC2=%.3fmA"%IC2);
print("IB1=%.2fμA"%IB1);
print("IB2=%.2fμA"%IB2);
print("VC1=%dV"%VC1);
print("VC2=%.1fV"%VC2);

#Variable declaration
VCC=15;                 #Collector supply voltage, V
VEE=15;                 #Emitter supply voltage, V
RB=33;                  #Base resistor, kΩ
RC=15;                  #Collector resistor, kΩ
RE=15;                  #Emitter resistor, kΩ
VBE=0.7;                #Base-emitter voltage, V


#Calculation
IE_tail=(VEE-VBE)/RE;            #Tail current, mA
IE=round(IE_tail/2,3);                    #Emitter current in each transistor, mA
IC=IE;                           #Collector current(=emitter current), mA
Vout=VCC-IC*RC;                  #Output voltage, V

#Result
print("The output voltage=%.2fV."%Vout);

#Variable declaration
VCC=15.0;                 #Collector supply voltage, V
VEE=15.0;                 #Emitter supply voltage, V
RB=33.0;                  #Base resistor, kΩ
RC=15.0;                  #Collector resistor, kΩ
RE=15.0;                  #Emitter resistor, kΩ
VBE=0;                  #Base-emitter voltage, V
beta_dc_l=90.0;           #base current amplification factor for left transistor
beta_dc_r=110.0;          #base current amplification factor for right transistor


#Calculation
#(i)
IE_tail=(VEE-VBE)/RE;             #Tail current, mA
IE=IE_tail/2;                      #Emitter current in each transistor, mA
IB1=(IE/beta_dc_l)*1000;          #Base current of 1st transistor, μA
IB2=(IE/beta_dc_r)*1000;          #Base current of 2nd transistor, μA

#(ii)
VB1=-IB1/1000*RB;                    #Base voltage of 1st transistor, V
VB2=-IB2/1000*RB;                    #Base voltage of 1st transistor, V

#Result
print("(i) IB1=%.2fμA"%IB1);
print("    IB2=%.2fμA"%IB2);
print("(ii) VB1=%.3fV"%VB1);
print("     VB2=%.2fV"%VB2);

#Variable declaration
VCC=15.0;                 #Collector supply voltage, V
VEE=15.0;                 #Emitter supply voltage, V
RB=10.0;                  #Base resistor, kΩ
RC1=10.0;                 #Collector resistor of 1st transistor, kΩ
RC2=10.0;                 #Collector resistor of 2nd transistor, kΩ
IE=1.0;                   #Tail current, mA
VBE=0.7;                #Base-emitter voltage, V


#Calculation
VE=-VBE;                    #Emitter voltage, V (Ignoring the base current)
IE1=IE/2.0;                   #Emitter current of 1st transistor, mA
IE2=IE1;                    #Emitter current of 2nd transistor, mA
IC1=IE1;                    #Collector current(= emitter current) of 1st transistor, mA
IC2=IE2;                    #Collector current of 2nd transistor, mA
VC1=VCC-IC1*RC1;            #Collector voltage of 1st transistor, V
VC2=VCC-IC2*RC2;            #Collector voltage of 2nd transistor, V


#Result
print("VE=%.1fV"%VE);
print("Emitter current in each transistor=%.1fmA."%(IE/2.0));
print("IC1~IE1=%.1fmA and IC2~IE2=%.1fmA"%(IE1,IE2));
print("VC1=VC2=%dV."%VC2);

#Variable declaration
VCC=12.0;                 #Collector supply voltage, V
VEE=12.0;                 #Emitter supply voltage, V
RC2=10.0;                 #Collector resistor of 2nd transistor, kΩ
RE=25.0;                  #Emitter current, kΩ
VBE=-0.7;               #Base-emitter voltage, V


#Calculation
VE=-VBE;                    #Emitter voltage, V (Ignoring the base current)
IE=(VCC-VE)/RE;             #Tail current, mA
IE1=IE/2.0;                   #Emitter current of 1st transistor, mA
IE2=IE1;                    #Emitter current of 2nd transistor, mA
IC1=IE1;                    #Collector current(= emitter current) of 1st transistor, mA
IC2=IE2;                    #Collector current of 2nd transistor, mA
VC1=-VEE;                   #Collector voltage of 1st transistor, V
VC2=-VEE+IC2*RC2;            #Collector voltage of 2nd transistor, V


#Result
print("VE=%.1fV"%VE);
print("Tail current=%.3fmA."%IE);
print("Emitter current in each transistor=%.3fmA."%(IE/2.0));
print("IC1~IE1=%.3fmA and IC2~IE2=%.3fmA"%(IC1,IC2));
print("VC1=%dV"%VC1);
print("VC2=%.2fV"%VC2);

#Variable declaration
VCC=15;                 #Collector supply voltage, V
VEE=15;                 #Emitter supply voltage, V
RB=1;                   #Base resistor, MΩ
RC2=1;                  #Collector resistor, MΩ
RE=1;                   #Emitter resistor, MΩ
VBE=0;                  #Base-emitter voltage, V (Neglected)
beta_dc_l=90.0;           #base current amplification factor for left transistor
beta_dc_r=110.0;          #base current amplification factor for right transistor


#Calculation
#(i)
IE=(VEE-VBE)/RE;                    #Tail current, μA
IE1=IE/2.0;                           #Emitter current of 1st transistor, μA
IE2=IE1;                            #Emitter current of 2nd transistor, μA
IB1=round((IE1/beta_dc_l)*1000,1);           #Base current of 1st transistor, nA
IB2=round((IE2/beta_dc_r)*1000,1);           #Base current of 2nd transistor, nA
I_in_offset=IB1-IB2;                 #Input offset current, nA

#(ii)
I_in_bias=(IB1+IB2)/2;                  #Input bias current, nA

#Result
print("(i)  The input offset current=%.1fnA"%I_in_offset);
print("(ii) The input bias current=%.1fnA"%I_in_bias);

#Variable declaration
I_in_offset=20;                 #Input offset current, nA
I_in_bias=80;                   #Input bias current, nA

#Calculation
IB1=I_in_bias+I_in_offset/2;            #Base current in 1st transistor, nA
IB2=I_in_bias-I_in_offset/2;            #Base current in 2nd transistor, nA


#Result
print("The two base currents are: IB1=%dnA and IB2=%dnA."%(IB1,IB2));

#Variable declration
I_in_offset=20;                 #Input offset current, nA
I_in_bias=80;                   #Input bias current, nA
A=150;                          #Voltage gain
RB=100;                         #Base resistor, kΩ


#Calculation
V_io=(I_in_offset*10**-9*RB*1000)*1000;       #Input offset voltage, mV
V_out_offset=(A*V_io)/1000;                     #Output offset voltage, V

#Result
print("The input offset voltage=%dmV."%V_io);
print("The output offset voltage=%.1fV."%V_out_offset);

#Variable declaration
VCC=15;                 #Collector supply voltage, V
VEE=15;                 #Emitter supply voltage, V
RE=1;                   #Emitter resistor, MΩ
RC=1;                   #Collector resistor, MΩ


#Calculation
IE=VEE/RE;                  #Tail current, μA
IE1=IE/2.0;                   #Emitter current of 1st transistor, μA
IE2=IE1;                    #Emitter current of 2nd transistor, μA
re=25/IE1;                  #a.c emitter resistance, kΩ
A_DM=RC/(2.0*re);        #Differential voltage gain,

#(i)
vin=1;                      #Input voltage, V
Vout=A_DM*vin;              #Output voltage, V

print("(i) Output voltage=%.2fV."%Vout);

#(ii)
vin=-1;                      #Input voltage, V
Vout=A_DM*vin;              #Output voltage, V;
print("(ii) Output voltage=%.2fV."%Vout);

#Variable declaration
VCC=12;                 #Collector supply voltage, V
VEE=12;                 #Emitter supply voltage, V
RE=100;                 #Emitter resistor, kΩ
RC1=120;                #Collector resistor of 1st transistor, kΩ
RC2=120;                #Collector resistor of 2nd transistor, kΩ
beta=220;               #Base amplification factor
VBE=0.7;                #Base-emitter voltage, V


#Calcualtion
IE=((VEE-VBE)/RE)*1000;     #Tail current, μA
IE1=IE/2.0;                   #Emitter current of 1st transistor, μA
IE2=IE1;                    #Emitter current of 2nd transistor, μA
re=(25/IE1)*1000;           #a.c emitter resistance, Ω
Zin=2*beta*re/1000;         #Input impedance, kΩ
A_DM=RC1*1000/(2.0*re);             #Differential voltage gain,


#Result
print("(i) The input impedance=%dkΩ."%Zin);
print("(ii) The differential voltage gain=%.0f."%A_DM);

#Variable declaration
VCC=12;                 #Collector supply voltage, V
VEE=12;                 #Emitter supply voltage, V
RE=200;                 #Emitter resistor, kΩ
RC=100;                 #Collector resistor, kΩ
VBE=0.7;                #Base-emitter voltage, V

#Calculation
IE=round((VEE-VBE)/RE,4);            #Tail current, mA
IE1=round(IE/2,4);                   #Emitter current of 1st transistor, mA
IE2=IE1;                             #Emitter current of 2nd transistor, mA
re=round(25/IE1,1);                  #a.c emitter resistance, Ω
A_DM=RC*1000/(2*re);                 #Differential voltage gain,

#Result
print("Differential voltage gain=%.1f."%A_DM);

from math import log10

#Variable declaration
v1=0.5;               #Voltage in terminal 1, mV
v2=-0.5;               #Voltage in terminal 2, mV
vo=8.0;                   #Output voltage, V
vo_cm=12.0;               #Common mode output, mV

#Calculation
vin=v1-v2;                  #Differential input, mV
A_DM=vo/(vin/1000.0);         #Differential mode gain,
vin_cm=1;                   #Common mode input, mV
A_CM=vo_cm/vin_cm;          #Common mode gain
CMRR=A_DM/A_CM;             #Common mode rejection ratio
CMRR_dB=20*log10(CMRR);     #Common mode rejection ratio in dB

#Result
print("Common mode rejection ratio=%.1f."%CMRR)
print("Common mode rejection ratio in decibel=%.2fdB"%CMRR_dB);

#Variable declaration
A_DM=200000;                #Differential mode gain
CMRR_dB=90;                 #Common mode rejection ratio, dB

#Calculation
CMRR=10**(CMRR_dB/20.0);              #Common mode rejection ratio
A_CM=A_DM/CMRR;                     #Common mode gain



#Result
print("Common mode voltage gain=%.2f."%A_CM);

from math import log10

#Variable declaration
vin_cm=3.2;                     #Common input voltage, V
vout=26;                        #Output voltage, V
A_DM=100;                       #Open-circuit voltage gain

#Calculation
#(i)
A_CM=vout*10**-3/vin_cm;                #Common mode gain

#(ii)
CMRR_dB=20*log10(A_DM/A_CM);            #Common mode rejection ratio, dB

#Result
print("(i)  The Common mode gain=%.4f"%A_CM);
print("(ii) The common mode rejection ratio=%.1fdB."%CMRR_dB);

from math import log10
from math import floor

#Variable declaration
VCC=12;                 #Collector supply voltage, V
VEE=12;                 #Emitter supply voltage, V
RE=200.0;                 #Emitter resistor, kΩ
RC=100.0;                 #Collector resistor, kΩ
VBE=0.7;                #Base-emitter voltage, V


#Calculation
#(i)
A_CM=round(RC/(2*RE),2);             #Common mode voltage gain

#(ii)
IE=round((VEE-VBE)/RE,4);            #Tail current, mA
IE1=round(IE/2,4);                   #Emitter current of 1st transistor, mA
IE2=IE1;                             #Emitter current of 2nd transistor, mA
re=round(25/IE1,1);                  #a.c emitter resistance, Ω
A_DM=RC*1000/(2*re);                 #Differential voltage gain,
CMRR_dB=floor(20*log10(A_DM/A_CM)*100)/100;         #Common mode rejection ratio, dB


#Result
print("(i) Common mode gain=%.2f"%A_CM);
print("(ii)Common mode rejection ratio=%.2fdB"%CMRR_dB);

from math import log10

#Variable declaration
ACL=500;                    #closed loop gain
f_unity=15;                 #frequency with cloased-loop unity gain, MHz


#Calculation
f2=f_unity*1000/500                 #Upper frequency of bandwidth,kHz
BW=f2-0;                            #Bandwidth, kHz
A_CL=f_unity*1000/200;                    #Maximum value of A_CL when f2=200kHz
A_CL_dB=20*log10(A_CL);             #Maximum value of A_CL in decibel


#Result
print("f2=%dkHz"%f2);
print("ACL=%d or %.1fdB."%(A_CL,A_CL_dB));


#Variable declaration
GBW=1.5;                    #Gain-bandwidth, MHz

#Calculation
#(i) For A_CL=1;
A_CL=1;                         #Closed loop gain
BW=GBW/A_CL;                    #Bandwidth, MHz

print("(i) Operating Bandwidth=%.1fMHz."%BW);

#(ii) For A_CL=10;
A_CL=10;                         #Closed loop gain
BW=(GBW/A_CL)*1000;              #Bandwidth, kHz

print("(ii) Operating Bandwidth=%dkHz."%BW);

#(iii) For A_CL=100;
A_CL=100;                         #Closed loop gain
BW=(GBW/A_CL)*1000;               #Bandwidth, kHz

print("(iii) Operating Bandwidth=%dkHz."%BW);

from math import pi

#Variable declaration
slew_rate=0.5;                  #Slew rate, V/μs
V_supply=10;                     #Supply voltage, V


#Calculation
V_sat=V_supply-2;                           #Saturation voltage, V
V_pk=V_sat;                                 #Maximum peak-output voltage, V
f_max=((slew_rate*10**6)/(2*pi*V_pk))/1000;       #Maximum operating frequency, kHz

#Result
print("Maximum operating frequency=%.2fkHz."%f_max);

from math import pi

#Variable declaration
slew_rate=0.5;                  #Slew rate, V/μs
V_pk=100.0;                       #Peak-output voltage, mV


#Calculation
V_pk=V_pk/1000.0;                                #Peak-output voltage, V
f_max=(slew_rate*10**6/(2*pi*V_pk))/1000.0;      #Maximum operating frequency, kHz

#Result
print("Maximum operating frequency=%.0fkHz"%f_max);

#Variable declaration
A_CL=-100;              #Closed-loop voltage gain
Ri=2.2;                 #Input resistor, kΩ

#Calculation
#Since, A_CL=-(Rf/Ri)
Rf=-A_CL*Ri;                #Feedback resistor, kΩ

#Result
print("Feedback resistor=%dkΩ"%Rf);

#Variable declaration
vin=2.5;                #Input voltage, mV
Rf=200;                 #Feedback resistor, kΩ
Ri=2;                   #Input resistor, kΩ

#Calculation
A_CL=-(Rf/Ri);                  #Closed-loop voltage gain
vout=A_CL*vin/1000;                  #Output voltage,V

#Result
print("Output voltage=%.2fV"%vout);

#Varaiable declaration
Rf=1.0;                   #Feedback resistor, kΩ
Ri=1.0;                   #Input resistor, kΩ

#Calculation
A_CL=-(Rf/Ri);                  #Closed-loop voltage gain

#Result
print("Closed-loop voltage gain=%d"%A_CL);
print("Therefore, output will have same amplitude but 180° phase inversion.");

#Variable declaration
Rf=40;                   #Feedback resistor, kΩ
Ri=1;                   #Input resistor, kΩ

#Calculation
A_CL=-(Rf/Ri);                  #Closed-loop voltage gain


#Result
print("Closed-loop voltage gain=%d"%A_CL);
print("Supply voltage=±15V, saturation voltage=±13V. Since gain=-40, op-Amp will be driven to saturation.");

from math import pi

#Variable declaration
Rf=100;                   #Feedback resistor, kΩ
Ri=10;                    #Input resistor, kΩ
Vpp=1;                    #Input peak-peak voltage, V
slew_rate=0.5;            #Slew rate, V//μs

#Calculation
#(i)
A_CL=-(Rf/Ri);                  #Closed-loop voltage gain

#(ii)
Zi=Ri;                  #Input impedance(~ Input resistor), kΩ

#(iii)
Vout=A_CL*Vpp;                           #Peak-to-peak voltage, V
Vpk=Vout/2;                              #Peak output voltage, V
f_max=(slew_rate*10**6/(2*pi*abs(Vpk)))/1000;        #Maximum operating frequency, kHz


#Result
print("(i)  A_CL=%d."%A_CL);
print("(ii) Zi=%dkΩ"%Zi);
print("(iii) Maximum operating frequency=%.1fkHz."%f_max);

#Variable declaration
A_CL=-4;                    #Closed loop voltage gain
R=[1.0,5.0,10.0,20.0];              #List of available resistors, kΩ

#Calculation
for i in R[:]:
    for j in R[:]:
        if -(i/j)==A_CL :
            print("Rf=%dkΩ and Ri=%dkΩ."%(i,j));
            break;


#Variable declaration
Rf=100;                   #Feedback resistor, kΩ
Ri=1;                    #Input resistor, kΩ


#Calculation
#(i)
R_source=0;               #Source resistor, kΩ
A_CL=-Rf/(R_source+Ri);    #Closed-loop voltage gain

print("(i) Closed loop voltage gain=%d."%A_CL);

#(ii)
R_source=1;               #Source resistor, kΩ
A_CL=-Rf/(R_source+Ri);    #Closed-loop voltage gain

print("(ii) Closed loop voltage gain=%d."%A_CL);

#Variable declaration
Rf=240;                   #Feedback resistor, kΩ
Ri=2.4;                    #Input resistor, kΩ
Vin=120;                    #Input voltage, μV


#Calculation
A_CL=1+(Rf/Ri);             #Closed loop voltage gain
Vout=(A_CL*Vin)/1000;       #Output voltage, mV

#Result
print("Output voltage=%.2fmV"%Vout);

#Variable declaration
Rf=10;                   #Feedback resistor, kΩ
Ri=1;                    #Input resistor, kΩ


#Calculation
A_CL=1+(Rf/Ri);             #Closed loop voltage gain
#(i)
Vin=1;                      #Input voltage, V
Vout=A_CL*Vin;              #Output voltage, V

print("(i)  Output voltage=%dV"%Vout);


#(ii)
Vin=-1;                      #Input voltage, V
Vout=A_CL*Vin;               #Output voltage, V

print("(ii) Output voltage=%dV"%Vout);

#Variable declaration
Rf=5;                       #Feedback resistor, kΩ
Ri=1;                       #Input resistor, kΩ
Vin_max=1;                  #Maximum input voltage, V
Vin_min=-1;                 #Minimum input voltage, V

#Calculation
V_inpp=Vin_max-Vin_min;               #Peak-peak input voltage, V
A_CL=1+(Rf/Ri);                     #Closed loop voltage gain
Vout_pp=A_CL*V_inpp;                #Peak-peak output voltage, V

#Result
print("Peak to peak output voltage=%dV"%Vout_pp);

from math import pi

#Variable declaration
Rf=100;                       #Feedback resistor, kΩ
Ri=10;                       #Input resistor, kΩ
Vpp=1;                    #Input peak-peak voltage, V
slew_rate=0.5;              #Slew rate, V/μs

#Calculation
#(i)
A_CL=1+(Rf/Ri);                                      #Closed loop voltage gain

#(ii)
Vout_pp=A_CL*Vpp;                                    #Peak-peak output voltage, V
Vpk=Vout_pp/2.0;                                          #Peak output voltage, V
f_max=((slew_rate*10**6)/(2*pi*Vpk))/1000.0;           #Maximum operating frequency, kHz

#Result
print("(i)  Closed-loop voltage gain=%d"%A_CL);

print("(ii) Maximum operating frequency=%.2fkHz"%f_max);

#Variable declaration
Rf=220;                       #Feedback resistor, kΩ
Ri=3.3;                       #Input resistor, kΩ
unity_gain_BW=3;              #Unity gain bandwidth, MHz

#Calculation
#(i) For non-inverting amplifier
A_CL=1+(Rf/Ri);                    #Closed loop voltage gain
BW=unity_gain_BW*1000.0/A_CL;        #Bandwidth, kHz

print("(i)  Bandwidth=%.1fkHz."%BW);

#(ii) For inverting amplifier
Rf=47;                          #Feedback resistor, kΩ
Ri=1;                           #Input resistor, kΩ
A_CL=-(Rf/Ri);                    #Closed loop voltage gain
BW=unity_gain_BW*1000.0/abs(A_CL);        #Bandwidth, kHz

print("(ii)  Bandwidth=%.1fkHz."%BW);

from math import pi

#(i)
A_CL=1;                     #Closed loop voltage gain for voltage follower
print("(i) For voltage follower A_CL=1.");


#(ii)
slew_rate=0.5;              #Slew rate, V/μs
V_inpp=6;                   #peak-peak input voltage, V
Vout=A_CL*V_inpp;           #Peak-peak output voltage, V
Vpk=Vout/2;                 #Peak output voltage, V

f_max=(slew_rate*10**6/(2*pi*Vpk))/1000;       #Maximum operating frequency, kHz

#Result
print("(ii) The maximum output frequency=%.2fkHz."%f_max);

#Variable declaration
Rf=470.0;                 #Feedback resistor, kΩ
R1=4.3;                 #Input resistor of 1st op-Amp, kΩ
R2=33.0;                 #Input resistor of 2nd op-Amp, kΩ
R3=33.0;                 #Input resistor of 3rd op-Amp, kΩ
Vin=80.0;                 #Input voltage, μV.

#Calculation
A1=1+Rf/R1;                 #Gain of first op-Amp
A2=-round(Rf/R2,1);         #Gain of second op-Amp
A3=-round(Rf/R3,1);         #Gain of third op-Amp
A=A1*A2*A3;                 #Overall gain
Vout=A*Vin*10**-6;          #Output voltage, V

#Result
print("Output voltage=%.2fV"%Vout);

#Variable declaration
A1=10;                  #Voltage gain of 1st op-Amp
A2=-18;                 #Voltage gain of 2nd op-Amp
A3=-27;                 #Voltage gain of 3rd op-Amp
Rf=270;                 #Feedback resistor, kΩ
Vin=150;                #Input voltage, μV  


#Calculation
R1=Rf/(A1-1);                   #Input resistor of 1st op-Amp, kΩ
R2=-Rf/A2;                      #Input resistr of 2nd op-Amp, kΩ
R3=-Rf/A3;                      #Input resistor of 3rd op-Amp, kΩ

A=A1*A2*A3;                     #overall gain,
Vout=Vin*10**-6*A;              #Output voltage, V


#Result
print("R1=%dkΩ, R2=%dkΩ and R3=%dkΩ."%(R1,R2,R3));
print("Output voltage=%.3fV."%Vout);

#Variable declaration
Rf=500;                 #Feedback resistor, kΩ
A1=-10;                 #Gain of 1st op-Amp
A2=-20;                 #Gain of 2nd op-Amp
A3=-50;                 #Gain of 3rd op-Amp

#Calculation
R1=-Rf/A1;                      #Input resistor of 1st op-Amp, kΩ
R2=-Rf/A2;                      #Input resistor of 2nd op-Amp, kΩ
R3=-Rf/A3;                      #Input resistor of 3rd op-Amp, kΩ


#Result
print("R1=%dkΩ, R2=%dkΩ and R3=%dkΩ."%(R1,R2,R3));

#Variable declaration
Zin=2.0;              #Input impedance of op-Amp, MΩ
Zout=75.0;            #Output impedance of op-Amp, Ω
A_OL=200000.0;        #Open-loop voltage gain
Rf=220.0;             #Feedback resistor, kΩ
Ri=10.0;              #Input resistor, kΩ


#Calculation
#(i)
mv=round(Ri/(Ri+Rf),3);       #Feedback fraction
Zin_NI=Zin*(1+(A_OL*mv));     #Input impedance, MΩ
Zout_NI=Zout/(1+A_OL*mv);     #Output impedance, Ω

#(ii)
A_CL=1+Rf/Ri;               #Closed loop voltage gain

#Result
print("(i) The input impedance=%dMΩ and output impedance=%.1eΩ."%(Zin_NI,Zout_NI));
print("(ii) The closed loop voltage gain=%d."%A_CL);

#Variable declaration
#For voltage follower,
mv=1.0;               #Feedback fraction
A_OL=200000.0;        #Open-loop voltage gain
Zin=2.0;              #Input impedance of op-Amp, MΩ
Zout=75.0;            #Output impedance of op-Amp, Ω


#Calculation
Zin_VF=Zin*(1+(A_OL*mv));     #Input impedance, MΩ
Zout_VF=round(round(Zout/(1+A_OL*mv),6),5);     #Output impedance, Ω

#Result
print("The input impedance=%dMΩ and output impedance=%.2fe-03Ω."%(Zin_VF,Zout_VF*1000));

#Variable declaration
Rf=100;             #Feedback resistor, kΩ
Ri=1.0;               #Input resistor, kΩ
Zin=4;              #Input impedance of op-Amp, MΩ
Zout=50;            #Output impedance of op-Amp, Ω


#Calculation
Zin_I=Ri;                   #Input impedance, kΩ
Zout_I=Zout;                #Output impedance, Ω
A_CL=-(Rf/Ri);              #Closed loop voltage gain


#Result
print("The input impedance=%dkΩ and output impedance=%dΩ."%(Zin_I,Zout_I));
print("Closed-loop voltage gain=%d"%A_CL);

#Variable declaration
Rf=10;               #Feedback resistor, kΩ
Ri=10;               #Input resistor, kΩ
V1=3;                #Input voltage 1st, V
V2=1;                #Input voltage 2nd, V
V3=8;                #Input voltage 3rd, V


#Calculation
#Since, Rf=Ri, Vout=-(Rf/Ri)*(V1+V2+V3)= -(V1+V2+V3);
Vout=-(V1+V2+V3);           #Output voltage, V

#Result
print("Output voltage=%dV."%Vout);

#Variable declaration
Rf=10;               #Feedback resistor, kΩ
R1=1;                #Input resistor for input 1, kΩ
R2=1;                #Input resistor for input 2, kΩ
V1=0.2;              #Input voltage 1st, V
V2=0.5;              #Input voltage 2nd, V


#Calculation
R=R1;                       #Input resistor(=R1 or R2), kΩ
Vout=-(Rf/R)*(V1+V2);       #Output voltage, V

#Result
print("Output voltage=%dV."%Vout);

#Variable declaration
Rf=1;                #Feedback resistor, kΩ
Ri=10.0;               #Input resistor, kΩ
V1=10;                #Input voltage 1st, V
V2=8.0;                #Input voltage 2nd, V
V3=7.0;                #Input voltage 3rd, V


#Calculation
#Since, Vout=-(Rf/Ri)*(V1+V2+V3);
Vout=-(Rf/Ri)*(V1+V2+V3);           #Output voltage, V

#Result
print("Output voltage=%.1fV."%Vout);

#Variable declaration
V1=0.6;                 #Input voltage to 1st input resistor, V
V2=-1.4;                #Input voltage to 2nd input resistor, V
Rf=200;                 #Feedback resistor, kΩ
R1=400;                 #Input resistor 1, kΩ
R2=100.0;                 #Input resistor 2, kΩ

#Calculation
Vout=-Rf*(V1/R1 +V2/R2);                 #Output voltage, V

#Result
print("Output voltage=%.1fV"%Vout);

#Variable declaration
Rf=1.0;                 #Feedback resistor, kΩ
R1=1.0;                 #Input resistor 1, kΩ
R2=2.0;                 #Input resistor 2, kΩ
R3=4.0;                 #Input resistor 3, kΩ


#Calculation
Rf_R1=Rf/R1;                #Ratio of feedback resistor and 1st input resistor
Rf_R2=Rf/R2;                #Ratio of feedback resistor and 2nd input resistor
Rf_R3=Rf/R3;                #Ratio of feedback resistor and 3rd input resistor

#(i) First input combination
V1=10;                                   #Input voltage to 1st input resistor, V
V2=0;                                    #Input voltage to 2nd input resistor, V
V3=10;                                   #Input voltage to 3rd input resistor, V
Vout=-(V1*Rf_R1 +V2*Rf_R2 +V3*Rf_R3);          #Output voltage, V
print("(i) The output voltage=%.1fV"%Vout);

#(i) First input combination
V1=0;                                   #Input voltage to 1st input resistor, V
V2=10;                                    #Input voltage to 2nd input resistor, V
V3=10;                                   #Input voltage to 3rd input resistor, V
Vout=-(V1*Rf_R1 +V2*Rf_R2 +V3*Rf_R3);          #Output voltage, V
print("(ii) The output voltage=%.1fV"%Vout);


#(i) First input combination
V1=10;                                   #Input voltage to 1st input resistor, V
V2=10;                                    #Input voltage to 2nd input resistor, V
V3=10;                                   #Input voltage to 3rd input resistor, V
Vout=-(V1*Rf_R1 +V2*Rf_R2 +V3*Rf_R3);          #Output voltage, V
print("(iii) The output voltage=%.1fV"%Vout);

#Variable declaration
Rf=330;                 #Feedback resistor, kΩ
R1=33.0;                  #Input resistor 1, kΩ
R2=10.0;                  #Input resistor 2, kΩ
V1_m=50;                #Peak voltage of 1st input, mV
V2_m=10;                #Peak voltage of 2nd input, mV

#Calculation
#Since, Vout=-((Rf/R1)*V1 + (Rf/R2)*V2)
print("Vout=-[%.1fsin(1000t)+%.2fsin(3000t)]V"%((V1_m/1000.0)*(Rf/R1),(V2_m/1000.0)*(Rf/R2)));

#Variable declaration
R=100;                      #Input resistor, kΩ
C=10;                       #Feedback capacitor, μF

#Calculation
RC=R*10**3*C*10**-6;        #product of input resistance and feedback capacitance, s


#Result
print("Vo=-1*(1/RC)∫vi dt.");
print("=>Vo=-1*(1/%d)∫vi dt"%RC);
print("=>Vo=∫vi dt");

from math import pi

#Variable declaration
Rf=100;                      #Feedback resistor, kΩ
C=0.01;                      #Feedback capacitor, μF


#Calculation
fc=1/(2*pi*Rf*1000*C*10**-6);               #Crictical frequency, Hz

#Result
print("The critical frequency=%dHz."%fc);

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#Variable declaration
R=10.0;                      #Input resistor, kΩ
C=0.01;                    #Feedback capacitor, μF
vin=5;                     #Input voltage, V

#Calculation
#(i)
Vout_change_rate=-vin/(R*C);            #Rate of change of output voltage, V/μs   
print("(i) Vout=-1*(1/RC)∫vi dt.");
print("    ΔVout/dt = -vin/RC = %dmV/μs."%Vout_change_rate);

#(ii) Plotting the output waveform
vin_plot=[];                    #Plotting variable for input waveform, V
dt=100;                          #time between edges, μs
for i in range(0,3*dt+1):
    if i<dt or i>2*dt :
        vin_plot.append(0);
    else:
        vin_plot.append(5); 

plt.subplot(211);
plt.plot(vin_plot);
plt.xlim([0,300])
plt.ylim([-5,10])
plt.xlabel("t(microsecond)");
plt.ylabel("Vin(V)");
plt.title("Input waveform");

        
vout_plot=[];                  #Plotting variable for output waveform, V
t=[i for i in range(0,301)];  #Time scale, μs
for i in t[:] :
    if i<dt:
        vout_plot.append(0);
    elif i>2*dt:
        vout_plot.append((Vout_change_rate/1000.0)*dt);
    else :
        vout_plot.append((-vin_plot[i]/(R*C))/1000*(i-dt));

plt.subplot(212)
plt.plot(vout_plot);
plt.xlim([0,300])
plt.ylim([-5,5]);
plt.xlabel('t(microsecond)');
plt.ylabel("Vout(V)");
plt.title("output waveform");

#Variable declaration
V_supply=15;              #Supply voltage, V
R=10;                     #Input resistor, kΩ
C=0.2;                    #Feedback capacitor, μF
vin=10;                   #Input voltage, mV


#Calculation
Vs=-V_supply+2;         #Saturation voltage, V
print("Vout=-1*(1/RC)∫vi dt.");
print("Vout=%d*t volts"%(-vin/(R*C)));
t=Vs/(-vin/(R*C));                      #Time required, seconds
print("Time required=%.1fseconds."%t);

#Variable declaration
R=1;                     #Feedback resistor, kΩ
C=0.1;                   #Input capacitor, μF
Vin_change=5;            #Change in input voltage, V
t=0.1;                   #Time taken for change in input voltage, ms

#Calcualtion
dvi_dt=Vin_change/(t/1000);        #Rate of change of input voltage, V/s
RC=R*1000*C*10**-6;                #Product of feedback resistance and input capacitance, s
#Since, Vo=-R*C*(dvi/dt);
Vo=-RC*dvi_dt;                      #Output voltage, V

#Result
print("Vo=%dV."%Vo);

#Variable declaration
R=10;                     #Feedback resistor, kΩ
C=2.2;                   #Input capacitor, μF
Vin_change=10;            #Change in input voltage, V
t=0.4;                   #Time taken for change in input voltage, s

#Calcualtin
dvi_dt=Vin_change/t;        #Rate of change of input voltage, V/s
RC=R*1000*C*10**-6;         #Product of feedback resistance and input capacitance, s
#Since, Vo=-R*C*(dvi/dt);
Vo=-RC*dvi_dt;                      #Output voltage, V

#Result
print("Vo=%.2fV."%Vo);
print("The output voltage stays constant at %.2fV."%Vo);

#Variable declaration
R=100;                     #Feedback resistor, kΩ
C=10;                   #Input capacitor, μF
Vin_change=1;            #Change in input voltage, V
t=0.2;                   #Time taken for change in input voltage, s


#Calculation
RC=R*1000*C*10**-6;             #Product of feedback resistor and input capacitance, s
#(i)
print("vo=-%d*(dvi/dt)."%RC);

#(ii)
dvi_dt=Vin_change/t;                #Rate of change of input voltage, V
vo=-dvi_dt;                         #Output voltage, V
print("vo=%dV."%vo);

print("Therefore, between 0 to 0.2s, the output voltage is constant at %dV."%vo);
print("For t>0.2s, the input is constant so that output voltage is zero.");



