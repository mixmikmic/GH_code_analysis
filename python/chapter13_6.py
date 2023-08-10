#Variable declaration
Av=3000.0;              #Voltage gain without feedback
m_v=0.01;               #Feedback fraction

#Calculation
#Since, Gain_with_feedback= Gain_without_feedback/(1+Gain_without_feedback*feedback_fraction),
Avf=Av/(1+Av*m_v);                  #Voltage gain of the amplifier with negative feedback

#Result
print("The voltage gain of the amplifier with negative feedback=%.0f."%Avf);

#Variable declaration
Av=140.0;                   #Voltage gain
Avf=17.5;                   #Voltage gain with negative feedback

#Calculation
#Since, Avf=Av/(1+Av*mv), so,
mv=(Av-Avf)/(Av*Avf);                 #Fraction of output fedback to the input


#Result
print("The fraction of output fedback to the input=1/%.0f."%(1.0/mv));

#Variable declaration
Av=100.0;                   #Voltage gain
Avf=50.0;                   #Voltage gain with negative feedback

#Calculation
#(i)
#Since, Gain_with_feedback= Gain_without_feedback/(1+Gain_without_feedback*feedback_fraction),
mv=(Av-Avf)/(Av*Avf);       #The fraction of output fedback to input

#(ii) Overall gain is to be 75:
Avf=75.0;                           #The required overall gain
#Since, Gain_with_feedback= Gain_without_feedback/(1+Gain_without_feedback*feedback_fraction),
Av=Avf/(1-Avf*mv);                  #The required value of amplifier gain

#result
print("(i)  The fraction of output fedback to input=%.2f."%mv);
print("(ii) The required amplifier gain for overall gain to be 75=%d."%Av);

#Variable declaration
Vout=10.0;              #output voltage , V
Vin_f=0.5;              #Input votage for amplifier with feedback, V
Vin=0.25;                #Input votage for amplifier without feedback, V

#Calculation
#(i)
Av=Vout/Vin;                #Voltage gain without negative feedback

#(ii)
Avf=Vout/Vin_f;             #Voltage gain with negative feedback
#Since, Gain_with_feedback= Gain_without_feedback/(1+Gain_without_feedback*feedback_fraction),
mv=(Av-Avf)/(Av*Avf);       #Feedback fraction

#Result
print("(i)  The voltage gain without feedback=%d."%Av);
print("(ii) The feedback fraction = 1/%d."%(1/mv));

#Variable declaration
Av=50.0;                #Gain without feedback
Avf=25.0;               #Gain with negative feedback

#Calculation
#Since, Gain_with_feedback= Gain_without_feedback/(1+Gain_without_feedback*feedback_fraction),
mv=(Av-Avf)/(Av*Avf);               #Feedback fraction

#(i)
#percentage of reduction without feedback
Av_reduced=40.0;                                        #Reduced amplifier gain due to ageing
percentage_of_reduction=((Av-Av_reduced)/Av)*100;       #Percentage of reduction in stage gain

print("(i)  The percentage of reduction in stage gain without feedback=%d%%."%percentage_of_reduction);

#(ii)
#Since, Gain_with_feedback= Gain_without_feedback/(1+Gain_without_feedback*feedback_fraction),
Avf_reduced=round(Av_reduced/(1+mv*Av_reduced),1);               #Reduced net gain with negative feedback 
percentage_of_reduction_f=((Avf-Avf_reduced)/Avf)*100;          #Percentage of reduction in net gain with feedback

print("(ii) The percentage of reduction in net gain with feedback=%.1f%%"%percentage_of_reduction_f);

#Variable declaration
Av=100.0;               #Gain
mv=0.1;                 #feedback fraction
Av_fall=6.0;            #fall in gain, dB

#Calculation
#Since, Gain_with_feedback= Gain_without_feedback/(1+Gain_without_feedback*feedback_fraction),
Avf=round(Av/(1+Av*mv),2);           #Total system gain with feedback

#Since, fall in gain=20*log10(Av/Av_1)
Av1=round(Av/10**(Av_fall/20),0);            #New absolute voltage gain without feedback
#Since, Gain_with_feedback= Gain_without_feedback/(1+Gain_without_feedback*feedback_fraction),
Avf_new=round(Av1/(1+Av1*mv),2);             #New net system gain with feedback

percentage_change=((Avf-Avf_new)/Avf)*100;          #Percentage change in system gain

#Result
print("The percentage change in system gain=%.2f%%"%percentage_change);

#Variable declaration
Av=500.0;                       #Voltage gain without feedback
Avf=100.0;                      #Voltage gain with negative feedback
Av_fall_percentage=20.0;          #Gain fall percentage due to ageing


#Calculation
#Since, Gain_with_feedback= Gain_without_feedback/(1+Gain_without_feedback*feedback_fraction),
mv=(Av-Avf)/(Av*Avf);                                           #Feedback fraction
Av_reduced=((100-Av_fall_percentage)/100)*Av;                   #Reduced voltage gain
Avf_reduced=round(Av_reduced/(1+Av_reduced*mv),1);                       #Reduced total gain of the system
percentage_fall=((Avf-Avf_reduced)/Avf)*100;                    #Percentage of fall in total system gain

#Result
print("The feedback fraction=%.3f."%mv);
print("The percentage fall in system gain=%.1f%%."%percentage_fall);

#Note: The percentage gain is calculated in the text as 4.7% due to approximation of Avf to 95.3 whose actual approximation will be (95.238)~95.2. So, the percentage fall calculated here is 4.8%

from math import log10

#Variable declaration
Av=100000.0;                        #Open loop voltage gain
f_dB=10.0;                          #Negative feedback, dB

#Calculation
Av_dB=20*log10(Av);                        #dB voltage gain without feedback, dB
Avf_dB=Av_dB-f_dB;                         #dB voltage gain with feedback, dB
Avf=10**(Avf_dB/20);                       #Voltage gain with feedback

#Since, Gain_with_feedback= Gain_without_feedback/(1+Gain_without_feedback*feedback_fraction),
mv=(Av-Avf)/(Av*Avf);                       #feedback fraction

#Result
print("The voltage gain with feedback=%d."%Avf);
print("The feedback fraction=%.2e."%mv);

#Variable declaration
Ao=1000.0;                        #Open circuit voltage gain
Rout=100.0;                       #Output resistance, ohm
RL=900.0;                         #Resistive load, ohm
mv=1/50;                          #feedback fraction

#Calculation
#Since, Av=Ao*RL/(Rout+RL)
Av=Ao*RL/(Rout+RL);                     #Voltage gain without feedback
Avf=Av/(1+Av*mv);                       #Voltage gain with feedback

#Result
print("The voltage gain with feedback=%.1f."%Avf);

#Variable declaration
Avf=100.0;                    #Voltage gain with feedback
vary_f=1;                     #Vary percentage in voltage gain with feedback
vary_wf=20;                   #Vary percentage in voltage gain without feedback

#Calculation
#Avf=Av/(1+Av*mv)
print("%d=Av/(1+Av*mv)        ------Eq. 1"%Avf);         #Equation 1

#considering variation in gains
Avf_vary=Avf*(1- vary_f/100.0);               #Gain with feedback, considering variation
print("%d=%.1f*Av/(1+%.1f*Av*mv)        ------Eq. 2"%(Avf_vary,(1-vary_wf/100.0),(1-vary_wf/100.0)));     #Equation 2

#Solving the above two equations
print("%d + %.1f*Av*mv=%.1fAv        ------Eq. 3 from Eq. 2"%(Avf_vary,Avf_vary*(1-vary_wf/100.0),(1-vary_wf/100.0)));     #Equation 3

#multiplying Eq. 1 with (Avf_vary*(1-vary_wf/100.0))/100=0.792
print("%.1f + %.1f*Av*mv=%.3fAv        ------Eq. 4 from Eq. 1"%(Avf*Avf_vary*(1-vary_wf/100.0)/100.0,Avf*Avf_vary*(1-vary_wf/100.0)/100.0,Avf_vary*(1-vary_wf/100.0)/100.0));     #Equation 4

print("Subtracting Eq.4 from Eq.3" );
print("%.1f = %.3f*Av"%(Avf_vary-Avf*Avf_vary*(1-vary_wf/100.0)/100.0,(1-vary_wf/100.0)-Avf_vary*(1-vary_wf/100.0)/100.0));
Av=(Avf_vary-Avf*Avf_vary*(1-vary_wf/100.0)/100.0)/((1-vary_wf/100.0)-Avf_vary*(1-vary_wf/100.0)/100.0);
print("Av=%.0f."%Av);
mv=(Av-Avf)/(Av*Avf);
print("mv=%.4f."%mv);
      

#Variable declaration
Av=10000.0;             #Volage gain without feedback
R1=2.0;                 #Resistor R1, kilo ohm
R2=18.0;                #Resistor R2, kilo ohm
Vin=1.0;                #input voltage, mV

#Calculation
#(i)
mv=R1/(R1+R2);              #feedback fraction

#(ii)
#Since, Gain_with_feedback= Gain_without_feedback/(1+Gain_without_feedback*feedback_fraction),
Avf=round(Av/(1+Av*mv),0);           #Voltage gain with feedback

#(iii)
Vout=Avf*Vin;               #Output voltage, mV

#Result
print("(i)   Feedback fraction=%.1f."%mv);
print("(ii)  Voltage gain with feedback=%d."%Avf);
print("(iii) Output voltage=%dmV."%Vout);

#Variable declaration
Av=10000.0;             #Volateg gain without feedback
Zin=10.0;               #Input impedance, kilo ohm
Zout=100.0;             #Output impedance, ohm
R1=10.0;                #Resistor R1, kilo ohm
R2=90.0;                #Resistor R2, kilo ohm

#Calculation
#(i)
mv=R1/(R1+R2);                  #Feedback fraction

#(ii)
#Since, Gain_with_feedback= Gain_without_feedback/(1+Gain_without_feedback*feedback_fraction),
Avf=round(Av/(1+Av*mv),0);               #Voltage gain with feedback

#(iii)
Zin_feedback=((1+Av*mv)*Zin)/1000;             #Increased input impedance due to negative feedback, mega ohm

#(iv)
Zout_feedback=Zout/(1+Av*mv);             #Decreased output impedance due to negative feedback, ohm


#Result
print("(i)   Feedback fraction=%.1f."%mv);
print("(ii)  The voltage gain with feedback=%d."%Avf);
print("(iii) Increased input impedance due to negative feedback=%.0f mega ohm"%Zin_feedback);
print("(iv)  Decreased output impedance due to negative feedback=%.1f ohm."%Zout_feedback);

#Variable declaration
Av=150.0;                       #Voltage gain
D=5/100.0;                        #Distortion
mv=10/100.0;                      #Feedback fraction

#Calculation
Dvf=round((D/(1+Av*mv))*100,3);                #Distortion with negative feedback


#Result
print("Distortion with negative feedback=%.3f%%"%Dvf);

#Note: In the text, value of Dvf=0.3125% has been approximated to 0.313%. But, here the approximation is done to 0.312%

#Variable declaration
Av=1000.0;                      #Voltage gain
f1=1.5;                         #Lower cut-off frequency, kHz
f2=501.5;                       #Upper cut-off frequency, kHz
mv=1/100.0;                       #Feedbcack fraction

#Calculation
f1_f=(f1/(1+mv*Av))*1000;              #New lower cut-off frequency, Hz
f2_f=(f2*(1+mv*Av))/1000;              #New upper cut-off frequency, MHz


#Result
print("The new lower cut-off frequency=%.1fHz"%f1_f);
print("The new upper cut-off frequency=%.2fMHz"%f2_f);

#Variable declaration
Ai=200.0;                       #Current gain without feedback
mi=0.012;                       #Current attenuation

#Calculation
Aif=Ai/(1+Ai*mi);

#Result
print("The effective current gain=%.2f."%Aif);

#Variable declaration
Ai=240.0;                       #Current gain
Zin=15.0;                       #Input impedance without feedback, kilo ohm
mi=0.015;                       #Current feedback fraction

#Calculations
Zin_f=Zin/(1+mi*Ai);                #Input impedance with feedback, kilo ohm


#Result
print("The input impedance when negative feedback is applied=%.2f kilo ohm"%Zin_f);

#Variable declaration
Ai=200.0;                   #Current gain without feedback
Zout=3.0;                   #Output impedance without feedback, kilo ohm
mi=0.01;                    #current feedback fraction

#Calculation
Zout_f=Zout*(1+mi*Ai);              #Output impedance with negative feedback, kilo ohm

#Result
print("The output impedance with negative feedback=%dkilo ohm."%Zout_f);

#Variable declaration
Ai=250.0;                   #Current gain without feedback
BW=400.0;                   #Bandwidth, kHz
mi=0.01;                    #current feedback fraction

#Calculation
BW_f=BW*(1+mi*Ai);              #Bandwidth when negative feedback is applied, kHz

#Result
print("Bandwidth when negative feedback is applied=%dkHz."%BW_f);

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as p

#Variable declaration
VCC=18.0;                   #Supply voltage, V
R1=16.0;                    #Resistor R1, kilo ohm
R2=22.0;                    #Resistor R2, kilo ohm
RE=910.0;                   #Emitter resistor, ohm
VBE=0.7;                    #Base-emitter voltage, V

#Calculations
V2=VCC*R2/(R1+R2);                 #Voltage across R2, V (Voltage divider rule)
VE=V2-VBE;                         #Emitter voltage, V
IE=(VE/RE)*1000;                   #Emitter current, mA (OHM's LAW)

#D.C load line
IC_sat=(VCC/RE)*1000;                      #Collector saturation current, mA
VCE_off=VCC;                               #Collector-emitter voltage in off state, V

#Result
print("Value of VE=%.2fV and IE=%.2fmA"%(VE,IE));

#Plotting
VCE_plot=[0,VCE_off];            #Plotting variable for VCE
IC_plot=[IC_sat,0];              #Plotting variable for IC
p.plot(VCE_plot,IC_plot);
p.xlim(0,20)
p.ylim(0,25)
p.xlabel('VCE(V)');
p.ylabel('IC(mA)');
p.title('d.c load line');
p.grid();

#Variable declaration
VCC=10.0;                   #Supply voltage, V
R1=10.0;                    #Resistor R1, kilo ohm
R2=10.0;                    #Resistor R2, kilo ohm
RE=5.0;                     #Emitter resistance, kilo ohm
VBE=0.7;                    #Base-emitter voltage, V


#Calculation
V2=VCC*R2/(R1+R2);                 #Voltage across R2, V (Voltage divider rule)
VE=V2-VBE;                         #Emitter voltage, V
IE=(VE/RE);                   #Emitter current, mA (OHM's LAW)
re=25/IE;                          #a.c emitter resistance, ohm
Av=RE*1000/(re+RE*1000);                     #Voltage gain


#Result
print("The voltage gain of the emitter follower circuit=%.3f."%Av);

#Variable declaration
RE=5.0;                 #Emitter resistance, kilo ohm
re=29.1;                #a.c emitter resistance, ohm
RL=5.0;                 #Load resistance, kilo ohm

#Calculation
RE_ac=(RE*RL)/(RE+RL);               #New effective value of emitter resistance, kilo ohm
Av=RE_ac*1000/(re+RE_ac*1000);     #Voltage gain

#Result
print("The voltage gain=%.3f"%Av);

def pr(r1,r2):                     #Function for calculating parallel resistance
    return (r1*r2)/(r1+r2);

#Variable declaration
VCC=10.0;                   #Supply voltage, V
R1=10.0;                    #Resistor R1, kilo ohm
R2=10.0;                    #Resistor R2, kilo ohm
RE=4.3;                     #Emitter resistor, kilo ohm
RL=10.0;                    #Load resistance, kilo ohm
VBE=0.7;                    #Base-emitter voltage, V
beta=200.0;                 #Base current amplification factor

#Calculation
V2=VCC*R2/(R1+R2);                 #Voltage across R2, V (Voltage divider rule)
VE=V2-VBE;                         #Emitter voltage, V
IE=(VE/RE);                        #Emitter current, mA (OHM's LAW)
re=25/IE;                          #a.c emitter resistance, ohm
RE_eff=pr(RE,RL);                  #Effective external emitter resistance, kilo ohm
Zin_base=beta*(re/1000+RE_eff);         #Input impedance of the base of the transistor, kilo ohm
Zin=pr(pr(R1,R2),Zin_base);        #Input impedance of emitter follower, kilo ohm
#Approximate value of input impedance taken as parallel resistance of R1 and R2 and ignoring Zin_base due to its relatively large value
Zin_approx=pr(R1,R2);              #Approximate input impedance, kilo ohm

#Result
print("The input impedance of the emitter follower =%.2f kilo ohm"%Zin);
print("The approximate value of the input impedance=%d kilo ohm"%Zin_approx);

def pr(r1,r2):                     #Function for calculating parallel resistance
    return (r1*r2)/(r1+r2);


#Variable declaration
re=20.0;                    #a.c emitter resistance, ohm
R1=3.0;                     #Resistor R1, kilo ohm
R2=4.7;                     #Resistor R2, kilo ohm
RS=600.0;                   #Source resistance, kilo ohm
beta=200.0;                 #Base current amplification factor

#Calculation
Rin_ac=pr(pr(R1,R2)*1000,RS);            #Input a.c resistance, ohm
Zout=re + Rin_ac/beta;                   #Output impedance, ohm

#Result
print("The output impedance=%.1f ohm"%Zout);

#Variable declaration
VCC=10.0;                    #Supply voltage, V
R1=120.0;                    #Resistor R1, kilo ohm
R2=120.0;                    #Resistor R2, kilo ohm
RE=3.3;                      #Emitter resistor, kilo ohm
VBE=0.7;                     #Base-emitter voltage, V
beta_1=70.0;                 #Base current amplification factor of 1st transistor
beta_2=70.0;                 #Base current amplification factor of 2nd transistor

#Calculation
#(i)
V2=VCC*R2/(R1+R2);                 #Voltage across R2, V (Voltage divider rule)
IE_2=(V2-2*VBE)/RE;                #Emitter current, mA (OHM's LAW)

#(ii)
Zin=(beta_1*beta_2*RE)/1000;               #Input impedance, mega ohm

#Result
print("(i)  d.c value of current in RE=%.2fmA"%IE_2);
print("(ii) Input impedance=%.2f mega ohm."%Zin);

#Variable declaration
VCC=12.0;                    #Supply voltage, V
R1=20.0;                     #Resistor R1, kilo ohm
R2=10.0;                     #Resistor R2, kilo ohm
RC=4.0;                      #Collector resistor, kilo ohm
RE=2.0;                      #Emitter resistor, kilo ohm
VBE=0.7;                     #Base-emitter voltage, V
beta=100.0;                  #Base current amplification factor of 1st transistor

#Calculation
#(i) D.C Bias levels
VB1=VCC*R2/(R1+R2);                 #Base voltage of 1st transistor, V (Voltage divider rule)
VE1=VB1-VBE;                        #Emitter voltage of 1st transistor, V
VB2=VE1;                            #Base voltage of 2nd transistor, V
VE2=VB2-VBE;                        #Emitter voltage of 2nd transistor, V
IE2=VE2/RE;                         #Emitter current of 2nd transistor, mA (OHM' LAW)
IE1=IE2/beta;                       #Emitter current of 1st transistor, mA (IE~IC=beta*IB, here IB2=IE1)

#(ii) A.C analysis
re1=25/IE1;                         #a.c emitter resistance of 1st transistor
re2=25/IE2;                         #a.c emitter resistance of 2nd transistor


#Result
print("(i) D.C Bias levels: \n VB1= %dV, VE1=%.1fV, VB2=%.1fV, VE2=%.1fV, IE2=%.1fmA and IE1=%.3fmA."%(VB1,VE1,VB2,VE2,IE2,IE1));
print("(ii) A.C Analysis:   \n re1=%d ohm and re2=%.2f ohm "%(re1,re2));



