#Variable declaration
V_CC=6.0;                     #Collector supply voltage
R_C=2.5;                      #Collector load in kΩ

#Calculations

#(i)
#For faithful amplification Vce (collector-emitter voltage)> 1V for Si transistor.
V_CE_max=1;                                   #Maximum allowed collector-emitter voltage for faithful amplification, in V.
V_Rc_max=V_CC-V_CE_max;                       #maximum voltage drop across collector load in V.
I_C_max=V_Rc_max/R_C;                         #Maximum allowed collector current in mA

#(ii)
IC_min_zero_signal=I_C_max/2;                            #Minimum zero signal collector current in mA

#Results
print("The maximum allowed collector current during application of signal for faithful amplification = %d mA."%I_C_max);
print("The minimum zero signal collector current required = %d mA."%IC_min_zero_signal);

#Variable declaration
VCC=13.0;                     #Collector supply voltage in V
V_knee=1.0;                   #Knee voltage in V
R_C=4.0;                      #Collector load  in kΩ
rate_IC_VBE=5.0;              #Rate of change of collector current IC with base-emitter voltage VBE in mA/V.
beta=100.0;                   #base current amplification factor


#Calculations
V_Rc_max=VCC-V_knee;                    #Maximum allowed voltage across collector load in V
I_C_max=V_Rc_max/R_C;                   #Maximum allowed collector current in mA
I_B_max=I_C_max/beta;                   #Maximum base current in mA
I_B_max=I_B_max*1000;                   #Maximum base current in 𝜇A

V_B_max=I_C_max/rate_IC_VBE;            #Maximum base voltage signal in V
V_B_max=V_B_max*1000;                   #Maximum base voltage signal in mV

#Results
print("Maximum base current =%d 𝜇A."%I_B_max);
print("Maximum input signal voltage =%d mV."%V_B_max);

#Variable declaration
VCC=9.0;                  #Colector supply voltage in V
VBB=2.0;                  #Base supply voltage in V
R_B=100.0;                #Base resistor's resistance in kΩ
R_C=2.0;                  #Collector load in kΩ
beta=50.0;                #base current amplification factor

#Calculations

#Case (i):

#Applying Kirchhoff's law to the input circuit
#We get, IB*RB +VBE =VBB.
#Neglecting the small base-emitter voltage, we get:
I_B=VBB/R_B;                #Base current in mA
I_C=beta*I_B;               #Collector current in mA

print("Collector current = %dmA"%I_C);

#Applying Kirchhoff's law to the output ciruit
#We get, IC*RC + VCE= VCC.
#From the above equation, we get:
V_CE=VCC-I_C*R_C;               #Collector emitter voltage in V

print("Collector emitter voltage =%dV."%V_CE);


#Case (ii):

R_B=50.0;
I_B=VBB/R_B;
I_C=beta*I_B;
V_CE=VCC  - I_C*R_C;

print("The new operating point for base resistor RB=50 kΩ is, VCE=%dV and IC=%dmA."%(V_CE,I_C));

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#variable declaration
beta=100.0;               #base current amplification factor
VCC=6.0;                  #Collector suply voltagein V
VBE=0.7                   #Base emitter voltage in V
R_B=530.0;                #Base resistor's resistance in kΩ .
R_C=2.0;                  #Collector resistor's resistance in kΩ .

#Calculation
#D.C load line equation : VCE=VCC-IC*RC;
#Calculating maximum VCE  ,by IC=0;
I_C_Vce_max=0;                      #Collector current for maximum collector-emitter voltage, in mA
VCE_max=VCC;-I_C_Vce_max*R_C;       #Maximum collector-emitter voltage in V


#Calculating maximum collector current IC,by VCE=0;
V_CE_IC_max=0;             #Collector-emitter voltage for maximum collector current, in V         
I_C_max=(VCC-V_CE_IC_max)/R_C;        #Maximum collector current in mA


#Operating point:
#For input circuit, applying Kirchhoff's law, We get,
#VCC=IB*RB  + VBE.
#From the above equation,
IB=(VCC-VBE)/R_B;                #Base current in mA
IC=beta*IB;                    #Collector current

#From the output circuit, applying Kirchhoff's law, we get:
VCE=VCC-IC*R_C;                 #Collector-emitter voltage in V


#Stability factor
SF=beta+1;              

#Result
print("Operating point: VCE= %dV and IC=%d mA"%(VCE,IC));
print("Stability factor= %d."%SF);


#plot
limit = plt.gca()
limit.set_xlim([0,10])
limit.set_ylim([0,5])
VCE=[i for i in range(0,(int)(VCC+1))];          #Plot variable for V_CE
IC=[((VCC-i)/(R_C)) for i in (VCE[:])];      #Plot variable for I_C

p=plt.plot(VCE,IC);
plt.xlabel("VCE(V)");
plt.ylabel("IC(mA)");
plt.title("d.c load line");
plt.show(p);

#Variable declaration
VCC=12.0;                 #Collector supply voltage in V
beta=100.0;              #base current amplification factor
I_C_zero_signal=1.0;      #zero signal collector current in mA
VBE=0.3;                  #Base-emitter voltage of Ge transistor in V

#calculations

#Case(i)
I_B_zero_signal=I_C_zero_signal/beta;               #Zero signal base current in  mA

#applying the Kirchhoff's law along input circuit:
#We get, VCC=IB*RB +VBE
#From the above equation we get,
R_B=(VCC-VBE)/I_B_zero_signal;              #Required base resistor's resistance in kΩ

print("Value of base resistor for operating the given Ge transistor at zero signal IC=1mA is = %d kΩ"%R_B);



#Case(ii)
beta=50;
I_B=(VCC-VBE)/R_B;                  #Base current of another transistor with beta=50, in mA
I_C_zero_signal=beta*I_B;           #Zero signal collector current for beta=50 , in mA

print("The new value of zero signal collector current =%.1fmA"%I_C_zero_signal);

#Variable declaration
VCC=10.0;                     #Collector supply voltage in V
VBE=0;                        #Base emitter voltage in V(considering itas zero due to it's small value)
R_B=1.0;                      #Base resistor's resistance in MΩ
R_C=2.0;                      #Collector resistor's resistance in kΩ                  
R_E=1.0;                      #Emitter resistor's resistance in kΩ
beta=100.0;                   #Base current amplification factor

#Calculations
#using Kirchhoff's law in the input circuit, we get:
#VCC=IB*RB +VBE +IE*RE
#Since, IE=(beta +1)*I_B
#From the above equation we get:
I_B=round((VCC-VBE)/((beta + 1)*R_E + R_B*1000),4);           #Base current in mA
I_C=round(beta*I_B,2);                                   #Collector current in mA
I_E=I_B+I_C;                                    #Emitter current in mA

#Result
print("Base current =%.4f mA"%I_B);
print("Collector current =%.2f mA"%I_C);
print("Emitter current =%.3f mA"%I_E);

#Variable declaration
VCE=8.0;                      #Collector-emitter voltage at operating point in V
IC=2.0;                       #Colector current at operating point in mA
VCC=15.0;                     #Collector supply voltagein V
beta=100.0;                   #base current amplification factor
VBE=0.6;                    #base emitter voltage in V

#Calculations
#Applying Kirchhoff's law along the output circuit,
#we get, VCC=VCE+IC*RC.
#So, from above equation we get:
RC=(VCC-VCE)/IC;                 #Collector resistor's resistance in kΩ .
IB=IC/beta;                      #Base current in mA

#Applying Kirchhoff's law along the input circuit,
#we get, VCC=IB*RB + VBE
#So, from the above equation:
RB=(VCC-VBE)/IB;                #Base resistor's resistance in kΩ .


#Results
print("Collector load =%.1f kΩ ."%RC);
print("Base resistor=%d kΩ ."%RB);

#Variable declaration
VCC=12.0;                #Collector supply voltage in V
VBE=0.7;                 #Base-emitter voltage in V
RB=100.0;                #Base resistor's resistance in kΩ
RC=560.0;                #Collector resistor's resistance in Ω
beta_25=100.0;           #base current amplification factor at 25 degree celsius
beta_75=150.0;           #base current amplification factor at 25 degree celsius


#Calculations


#Applying Kirchhoff's law along input circuit, we get
#VCC=IB*RB+VBE
IB=(VCC-VBE)/RB;                #Base current at 25 degree celsius, in mA


#For temperature 25 degree celsius
IC_25=beta_25*IB;                  #Collector current at 25 degree celsius, in mA


#Applying Kirchhoff's alw at the output circuit,
#we get: VCC=IC*RC + VCE
#From the above equation,
VCE_25=round(VCC-(IC_25/1000)*RC,2);                #Collector emitter voltage  at 25 degree celsius, in V


#For temperature 75 degree celsius
IC_75=round(beta_75*IB,0);                  #Collector current at 75 degree celsius, in mA

#Applying Kirchhoff's alw at the output circuit,
#we get: VCC=IC*RC + VCE
#From the above equation,
VCE_75=round(VCC-(IC_75/1000)*RC,2);                #Collector emitter voltage at 75 degree celsius, in V


change_IC=(IC_75-IC_25)*100.0/IC_25;        #percentage change in collector current
change_VCE=(VCE_75-VCE_25)*100.0/VCE_25;    #Percentage change in collector-emitter voltage 

#Results
print("The percentage change in collector current =%d%%"%change_IC);
print("The percentage change in collector-emitter voltage =%.1f%%"%change_VCE);

#Variable declaration
VCE_max=20.0;             #Maximum collector-emitter voltage in V
VBE=0.7;                  #Base-emitter voltage in V
IC_max=8.0;               #Maximum collector current in mA
IB=40.0;                  #Base current in microampere

#Calculations

#During cut off state the collector-emitter voltage is maximum and equal to collector supply voltage
VCC=VCE_max;                #Collector supply voltage in V

#Maximum collector current IC_max=collector supply voltage(VCC)/collector load(RC)
#Collector load(RC)=VCC*IC_max
RC=VCC/IC_max;                  #Collector load in kΩ .

#Applying Kirchhoff's law along input circuit,
#we get, VCC=IB*RB +VBE.
#From the above equation, we get:
RB=(VCC-VBE)/(IB/1000);    #Base resistor's resistance in kΩ .

#Results
print("Collector supply voltage = %dV"%VCC);
print("Collector load=%.1f kΩ ."%RC);
print("Base resistor's resistance=%.1f kΩ ."%RB);

#Variable declaration
VCC=20.0;                     #Collector supply voltage in V
VEE=-20.0;                     #Emitter supply voltage in V
RB=100.0;                     #Base resistor's resistance in kΩ
RC=4.7;                       #Collector resistor's resistance in kΩ
RE=10.0;                      #Emitter resistor's resistance in kΩ
VBE=0.7;                      #Base-emitter voltage in V
beta=85.0;                    #Base current amplification factor


#Calculations
#Applying Kirchhoff's voltage law along the base-emitter circuit (input circuit),
#we get,IB*RB  +IE*RE +VBE -VEE=0.
#Since IB=IC/beta and IC~IE,
#(IE/beta)*RB + IE*RE + VBE + VEE =0.
IE=(-VEE-VBE)/(RE + RB/beta);                        #Emitter current in mA
IC=IE;                                              #Collector current (approximately equal to emitter current) in mA

#Applying Kirchhoff's law from VCC till collector terminal,
#we get, VCC - IC*RC =VC
VC=VCC-IC*RC;                               #voltage at collector terminal in V

#Applying Kirchhoff's law from emitter terminal to VEE
#we get, VE -IE*RE =VEE
VE=VEE + IE*RE;                             #Voltage at emitter treminal in V

VCE=VC-VE;                                  #Collector-emitter voltage in V

#Results
print("The collector current = %.2f mA"%IC);
print("The emitter current = %.2f mA"%IE);
print("The voltage at collector terminal = %.1f V"%VC);
print("The collector-emitter voltage = %.1f V"%VCE);

#Variable declaration
VCC=20.0;                     #Collector supply voltage in V
VEE=-20.0;                     #Emitter supply voltage in V
RB=100.0;                     #Base resistor's resistance in kΩ
RC=4.7;                       #Collector resistor's resistance in kΩ
RE=10.0;                      #Emitter resistor's resistance in kΩ
beta1=85.0;                   #Base current amplification factor for case 1       
beta2=100.0;                  #Base current amplification factor for case 1
VBE_1=0.7;                    #Base emitter voltage for case 1 in V
VBE_2=0.6;                    #Base emitter voltage for case 2 in V


#Calculations
#For beta=85 and VBE=0.7,
#As calculated in the previous question,
IC_1=1.73;                          #Collector current in mA.
VCE_1=14.6;                           #Collector-emitter voltage in V.


#For case (ii)
#beta=100 and VBE=0.6

#Applying Kirchhoff's voltage law along the base-emitter circuit (input circuit),
#we get,IB*RB  +IE*RE +VBE -VEE=0.
#Since IB=IC/beta and IC~IE,
#(IE/beta)*RB + IE*RE + VBE +VEE =0.
IE_2=round((-VEE-VBE_2)/(RE + RB/beta2),2);                        #Emitter current in mA
IC_2=IE_2;                                                #Collector current (approximately equal to emitter current) in mA

#Applying Kirchhoff's law from VCC till collector terminal,
#we get, VCC - IC*RC =VC
VC=round(VCC-IC_2*RC,1);                               #voltage at collector terminal in V

#Applying Kirchhoff's law from emitter terminal to VEE
#we get, VE -IE*RE =VEE
VE=round(VEE + IE_2*RE,1);                             #Voltage at emitter treminal in V

VCE_2=VC-VE;                                #Collector-emitter voltage in V


change_IC= (IC_2-IC_1)*100/IC_1;                    #%age change in collector current

change_VCE=(VCE_2-VCE_1)*100/VCE_2;                 #%age change in collector-emitter voltage



#Results
print("Percentage change in collector current =%.1f%%"%change_IC);
print("Percentage change in collector-emitter voltage =%.1f%%"%change_VCE);

#Variable declaration
VCC=20.0;                 #Collector supply voltage in V
VBE=0.7                   #Base-emitter voltage in V
RB=100.0;                 #Base resistor's resistance in kΩ
RC=1.0;                   #Collector resistor's resistance in kΩ
beta=100.0;               #base current amplification factor


#Calculations
#Applying Kirchhoff's law along input circuit,
#we get, VCC -IC*RC -IB*RB -VBE=0.
#since IC= beta*IB,
#We get,
IB=(VCC-VBE)/(RB + beta*RC);            #Base current in mA
IC=beta*IB;                             #Collector current in mA

#Applying Kirchhoff's law along the output circuit,
#we get, VCC-VCE - IC*RC=0.
#From the above equation,
VCE=VCC-IC*RC;                          #Collector emitter voltage in V


#Results
print("The operating point : VCE=%.2fV and IC=%.2fmA."%(VCE,IC));


#Variable declaration
VCC=12.0;                         #Collector supply voltage in V
VBE=0.3;                          #Base emitter voltage in V
IC=1.0;                           #Collector current in mA
VCE=8.0;                          #Collector emitter voltage in V
beta=100.0;                         #Base current amplification factor

#Calculations

#Case(i)

#Applying Kirchhoff's law along the output circuit,
#we get, VCC-IC*RC-VCE=0.
#from the above equation we get,
RC=(VCC-VCE)/IC;             #Collector load in kilo ohm
IB=IC/beta;                  #Base current in mA

#Applying Kirchhoff's law along input circuit
#we get, VCC-VBE-(beta*IB*RC)-IB*RB=0.
#From the above equation we get,
RB=round((VCC-VBE-beta*IB*RC)/IB,0);         #Base resistor's resistance in kΩ

#Results
print("The resistance value of base resistor=%d kΩ and collector load= %d kΩ."%(RB,RC));

#Case(ii)

beta=50;

#Applying Kirchhoff's law along input circuit,
#we get, VCC -IC*RC -IB*RB -VBE=0.
#since IC= beta*IB,
#We get,
IB=(VCC-VBE)/(RB + beta*RC);            #Base current in mA
IC=beta*IB;                             #Collector current in mA

#Applying Kirchhoff's law along the output circuit,
#we get, VCC-VCE - IC*RC=0.
#From the above equation,
VCE=round(VCC-IC*RC,1);                          #Collector emitter voltage in V

#Results
print("The operating point : VCE=%.1fV and IC=%.1fmA."%(VCE,IC));


#Variable declaration
VCE=2.0;                      #Collector-emitter voltage at operating point in V
VBE=0.7;                      #Base-emitter voltage in V            
IC=1.0;                       #Collector current at operating point in mA
beta=100.0;                   #Base current amplification factor

#Calculations
IB=IC/beta;                     #Base current in mA

#As, VCE=VCB +VBE
#we get,
VCB=VCE-VBE;                    #Collector-base voltage in V
RB=VCB/IB;                      #Base resistor's resistance in kΩ

#Results
print("Value of base resistor's resistance=%d kΩ."%RB);

#Variable declaration
VCC=12.0;                 #Collector supply voltage in V
VBE=0.7                   #Base-emitter voltage in V
RB=400.0;                 #Base resistor's resistance in kΩ
RC=4.0;                   #Collector resistor's resistance in kΩ
RE=1.0;                   #Emitter resistor's resistance in kΩ
beta=100.0;               #Base current amplification factor


#Calculations
#Applying Kirchhoff's law along outut circuit,
#we get, VCC -(IC+IB)*RC -IB*RB -VBE - IE*RE=0.
#since IC= beta*IB, IC+IB ~ IC and IE~IC,
#We get, VCC - IC*RC -(IC/beta)*RB -VBE - IE*RE
IC=(VCC-VBE)/(RB/beta + RC + RE);       #Collector current current in mA.
IE=IC;                                                #Emitter current in mA

#Applying Kirchhoff's law along the output circuit,
#we get, VCC-VCE - IC*RC -IE*RE=0. (IE~IC)
#From the above equation,
VCE=VCC-IC*(RC+RE);                          #Collector emitter voltage in V


#Results
print("The operating point : VCE=%.1fV and IC=%.2fmA."%(VCE,IC));


#Variable declaration
VCC=10.0;                     #Collector supply voltage in V
RB=100.0;                     #Base resistor's resistance in kΩ
RC=10.0;                      #Collector resistor's resistance in kΩ
RE=0;                         #Emitter resistor's  resistance in kΩ
VBE=0.7;                      #Base-emitter voltage in V
beta=100.0;                   #Base current amplification factor

#Calculations
#Applying Kirchhoff's law along outut circuit,
#we get, VCC -(IC+IB)*RC -IB*RB -VBE - IE*RE=0.
#since IC= beta*IB, IC+IB ~ IC and IE~IC,
#We get, VCC - IC*RC -(IC/beta)*RB -VBE - IE*RE
IC=(VCC-VBE)/(RC +RB/beta + RE);                #Collector current in mA

#Applying Kirchhoff's law along the output circuit,
#we get, VCC-VCE - IC*RC =0. (IE~IC)
#From the above equation,
VCE=VCC-IC*RC;                                  #Collector-emitter voltage in V

#Results
print("The d.c bias values are: VCE=%.2fV and IC=%.3fmA"%(VCE,IC));

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#Variable declaration
VCC=15.0;                           #Collector supply voltage in V
R1=10.0;                            #Resistor R1's resistance in kΩ
R2=5.0;                             #Resistor R2's resistance in kΩ
RC=1.0;                             #Collector resistor's resistance in kΩ              
RE=2.0;                             #Emitter resistor's resistance in kΩ
VBE=0.7;                            #Base-emitter voltage in V

#Calculations
#Applying Kirchhoff's law along output circuit
#VCE=VCC-IC*(RC+RE);
#IC=0, for VCE_max
VCE_max=VCC;                        #Maximum collector-emitter voltage in V
#VCE=0, for IC_max
IC_max=VCC/(RC+RE);                 #Maximum collector current in mA

#Operating point
V2=(VCC*R2)/(R1+R2);                #Voltage across R2 resistor V
IE=(V2-VBE)/RE;                     #Emitter current in mA
IC=IE;                              #Collector current(Approx. equal to emitter current) in mA
VCE=VCC-IC*(RC+RE);                 #Collector-emitter voltage in V

#Results
print("Collector-emitter voltage at operating point=%.2fV"%VCE);
print("Collector current at operating point = %.2fmA"%IC);

#plot
limit = plt.gca()
limit.set_xlim([0,20])
limit.set_ylim([0,6])
VCE=[i for i in range(0,(int)(VCC+1))];          #Plot variable for V_CE
IC=[((VCC-i)/(RC+RE)) for i in (VCE[:])];      #Plot variable for I_C

p=plt.plot(VCE,IC);
plt.xlabel("VCE(V)");
plt.ylabel("IC(mA)");
plt.title("d.c load line");
plt.show(p);

#Variable declaration
VCC=15.0;                           #Collector supply voltage in V
R1=10.0;                            #Resistor R1's resistance in kΩ .
R2=5.0;                             #Resistor R2's resistance in kΩ .
RC=1.0;                             #Collector resistor's resistance in kΩ .              
RE=2.0;                             #Emitter resistor's resistance in kΩ .
VBE=0.7;                            #Base-emitter voltage in V

#Calculations
#Using Thevenin's Theorem for replacing circuit consisting of VCC,R1,R2
E0=(VCC*R2)/(R1+R2);                #Thevenin's voltage in V
R0=(R1*R2)/(R1+R2);                 #Thevenin's equivalent resistance in kΩ .

#Applying Kirchhoff' law along thevenin's equivalent circuit,
#E0=IB*R0+VBE+IE*RE;
#Since IE~IC and IC=beta*IB
#IC=(E0-VBE)/(R0/beta +RE);
IC=(E0-VBE)/RE;                         #(Since R0/beta << RE) collector current in mA
VCE=VCC-IC*(RC+RE);                     #Collector emitter voltage in V


#Results
print("Collector-emitter voltage at operating point=%.2fV"%VCE);
print("Collector current at operating point = %.2fmA"%IC);

#Variable declaration
VCC=12.0;                           #Collector  supply voltage in V
RE=1.0;                             #Emitter resistor, kΩ .
R1=50.0;                            #Resistor R1, kΩ .
R2=10.0;                            #Resistor R2, kΩ .


#Calculations

#(i)
VBE=0.1;                            #Base-emitter voltage in V
V2=(VCC*R2)/(R1+R2);                #Voltage drop across resistor R2, V 
IE=(V2-VBE)/RE;                     #Emitter current in mA

print("(i)Emitter current= %.1fmA"%IE);

#(ii)
VBE=0.3;                            #Base-emitter voltage in V
V2=(VCC*R2)/(R1+R2);                #Voltage drop across resistor R2, V
IE=(V2-VBE)/RE;                     #Emitter current in mA

print("(ii)Emitter current= %.1fmA"%IE);

#Variable declaration
VCC=20.0;                   #Collector supply voltage, V
R1=10.0;                    #Resistor R1, kΩ
R2=10.0;                    #Resistor R2, kΩ .
RC=1.0;                     #Collector resistor, kΩ .
RE=5.0;                     #Emitter resistor, kΩ .


#Calculations
V2=(VCC*R2)/(R1+R2);                #Voltage drop across resistor R2, V

#Applying kirchhoff's law from base terminal to emitter resistor
#V2=VBE+IE*RE
#VBE is neglected due to its small value

IE=V2/RE;                     #Emitter current in mA
IC=IE;                              #Collector current (approx. equal to emitter current), mA

#Applying Kirchhoff's law along output circuit
VCE=VCC-IC*(RC+RE);                 #Collector-emitter voltage , V
VC=VCC-IC*RC;                       #Voltage at collector terminal,V


#Results
print("Emitter current =%dmA"%IE);
print("Collector-emitter voltage=%dV"%VCE);
print("Collector terminal's voltage=%dV"%VC);


#Variable declaration
VCC=12.0;                 #Collector supply voltage, V
VBE=0.7;                  #Base-emitter voltage, V
beta=50;                  #Base current amplification factor
R1=150;                   #Resistor R1, kΩ .
R2=100;                   #Resistor R2, kΩ .
RC=4.7;                   #Collector resistor, kΩ .
RE=2.2;                   #Emitter resistor, kΩ  .

#Calculations
#Using Thevenin's theorem, calculating Thevenin's voltage and resistance
E0=(VCC*R2)/(R1+R2);        #Thevenin's voltage, V
R0=(R1*R2)/(R1+R2);         #Thevenin's resistance, kΩ .

#Applying Kirchhoff' law along thevenin's equivalent circuit,
#E0=IB*R0+VBE+IE*RE;
#Since IE~IC and IC=beta*IB
IB=round((E0-VBE)/(R0+beta*RE),3);              #Base current in mA
IC=round(beta*IB,1);                             #Collector current, mA

#Applying Kirchhoff's law along the output circuit
VCE=VCC-IC*(RC+RE);                     #Collector-emitter voltage, V

S=(beta+1)*(1+R0/RE)/(beta +1+R0/RE);               #Stability factor


#Results
print("Operating point : VCE= %.2fV and IC=%.1fmA"%(VCE,IC));
print("Stability factor=%.1f"%S);


#Variable declaration
VCC=15.0;                 #Collector supply voltage, V
VBE=0.7;                  #Base-emitter voltage , V
beta=100.0;                 #Base current amplification factor
R1=6.0;                     #Resistor R1, kΩ .
R2=3.0;                     #Resistor R2, kΩ .
RC=470.0;                   #Collector resistor, Ω.
RE=1.0;                     #Emitter resistor, kΩ .

#Calculations
#Using Thevenin's theorem, calculating Thevenin's voltage and resistance
E0=(VCC*R2)/(R1+R2);        #Thevenin's voltage, V
R0=(R1*R2)/(R1+R2);         #Thevenin's resistance, kΩ .

#Applying Kirchhoff' law along thevenin's equivalent circuit,
#E0=IB*R0+VBE+IE*RE;
#Since IE~IC and IC=beta*IB
IB=round((E0-VBE)/(R0+beta*RE),3);              #Base current in mA
IC=round(beta*IB,1);                             #Collector current, mA

#Applying Kirchhoff's law along the output circuit
VCE=VCC-IC*(RC/1000+RE);                     #Collector-emitter voltage, V

S=(beta+1)*(1+R0/RE)/(beta +1+R0/RE);               #Stability factor

#Results
print("Operating point : VCE= %.2fV and IC=%.1fmA"%(VCE,IC));
print("Stability factor=%.2f"%S);


#Varaible declaration
VCC=9;                    #Collector supply voltage, V
VCE=3;                    #Collector-emitter voltage, V
VBE=0.3;                    #Base-emitter voltage in V
RC=2.2;                     #Collector resistor , kΩ .
IC=2;                     #Collector current, mA
beta=50.0;                  #Base current amplification factor

#Calculations
IB=IC/beta;                 #Base current in mA

#According to given relation, I1=10*IB
I1=IB*10;                   #Current through the resistor R1, mA

#I1=VCC/(R1+R2), .'s LAW
R1_R2_sum=VCC/I1;               #Sum of the resistor's R1 and R2, kΩ (OHM'S LAW).

#Applying Kirchhoff's law along the output circuit
#VCC=IC*RC+VCE+IE*RE
#IC~IE
RE=(VCC-IC*RC-VCE)/IC;      #Emitter resistor, kΩ .
RE=round(RE*1000,0);                 #Emittter resistor, Ω .

IE=IC;                      #Emittter current(approximately equal to collector current), mA
VE=IE*(RE/1000);                   #Voltage at emitter terminal (OHM's LAW), V
V2=VBE+VE;                  #Voltage drop across resistor R2, V

R2=V2/I1;                   #Resistor R2,(OHM's LAW), kΩ .
R1=R1_R2_sum-R2;            #Resistor R1, kΩ .



#Results
print("RE=%d Ω., R1=%.2f kΩ . and R2=%.2f kΩ ."%(RE,R1,R2));

#Variable declaration
VCC=16.0;                     #Collector supply voltage, V
R2=20.0;                      #Resistor R2, kΩ
RE=2.0;                       #Emitter resistor, kΩ
VCE=6.0;                      #Collector-emitter voltage, V
IC=2.0;                       #Collector current , mA
VBE=0.3;                      #Base-emitter voltage,V
alpha=0.985;                  #Current amplification factor

#Calculations
beta=alpha/(1-alpha);               #Base current amplificatioon factor
IE=IC;                              #Emitter current, mA
IB=IC/beta;                         #Base current, mA
VE=IE*RE;                           #Emitter voltage,(OHM's LAW) V
V2=VBE+VE;                          #Voltage drop across resistor R2,(Kirchhoff's law) V
V_R1=VCC-V2;                        #Voltage drop across resistor R1, V
I1=V2/R2;                           #Current through resistor R2 an R1,(OHM's LAW) mA
R1=V_R1/I1;                         #Resistor R1,(OHM's LAW) kΩ

V_RC=(VCC-VCE-VE);                  #Voltage across collector resistor, V
RC=V_RC/IC;                         #Collector resistor,(OHM's LAW) kΩ


#Results
print("R1=%.1f kΩ and RC=%d kΩ."%(R1,RC));

#Variable declaration
VCC=15.0;                     #Collector supply voltage, V
R1=10.0;                      #Resistor R1, kΩ 
R2=5.0;                       #Resistor R2, kΩ 
RC=1.0;                       #Collector resistor, kΩ 
RE=2.0;                       #Emitter resistor, kΩ 
VBE=0.7;                      #Base-emitter voltage, V
beta=100;                     #Base current amplification factor

#Calculations
#Using Thevenin's theorem, calculating Thevenin's voltage and resistance
E0=(VCC*R2)/(R1+R2);        #Thevenin's voltage, V
R0=(R1*R2)/(R1+R2);         #Thevenin's resistance, kΩ 

#Applying Kirchhoff' law along Thevenin's equivalent circuit,
#E0=IB*R0+VBE+IE*RE;
#Since IE~IC and IB=IE/beta,
IE=(E0-VBE)/(R0/beta + RE);                 #Emitter current , mA


#Calculations
print("The exact value of emitter current in the circuit = %.2fmA."%IE);


#Variable declaration
IE=2.0;                     #Emitter current, mA
IB=50.0;                    #Base current, mA
VCC=10.0;                   #Collector supply voltage, V
VBE=0.2;                    #Base-emitter voltage, V
R2=10.0;                    #Resistor R2, kΩ
RE=1.0;                     #Emitter resistance, kΩ

#Calculations
#Applying Kirchhoff's law from the base to the emitter resistor,
V2=VBE+IE*RE;                   #Voltage at base terminal, V
I2=V2/R2;                       #Current through the resistor R2, mA
I1=I2+IB/1000;                  #Current through the resistor R2, mA
V1=VCC-V2;                      #Voltage drop across the resistor R2
R1=V1/I1;                       #Resistor R1, kΩ


#Results
print("The value of the resistor R1=%.2f kΩ."%R1);

#Variable declaration
VCC=8.0;                  #Collector supply voltage, V
RB=360.0;                 #Base resistor, kΩ
RC=2.0;                   #Collector resistor, kΩ
VBE=0.7;                  #Base-emitter voltage, V
beta=100.0;               #base current amplification factor


#Calculations
IC_max=VCC/RC;              #Maximum collector current, mA
VCE_max=VCC;                #Maximum collector voltage, V

#Operating point
#Applying Kirchhoff's law along the input circuit
IB=(VCC-VBE)/RB;                    #Base current, mA
IC=beta*IB;                         #Collector current, mA

#Kirchhoff' law along the output circuit
VCE=VCC-IC*RC;                      #Collector-emitter voltage, V

#Results
print("VCE=%.2fV, is approximately half of VCC=%dV \n therefore it is mid-point biased."%(VCE,VCC));

#Variable declaration
VCC=10.0;                     #Collector supply voltage, V
VBE=0.7;                     #Base-emitter voltage, V
beta=50.0;                     #Base current amplification factor
R1=12.0;                     #Resistor R1, kΩ 
R2=2.7;                      #Resistor R2, kΩ 
RC=620.0;                    #Collector resistor, Ω 
RE=180.0;                    #Emitter resistor, Ω


#Calculations
#Voltage divder rule across R1 and R2
V2=round((VCC*R2)/(R1+R2),2);            #Voltage drop across resistor R2, V
IE=round(((V2-VBE)/RE)*1000,2);                 #Emitter current, mA
IC=IE;                          #Collector current(Approximately equal to emitter current), mA
print("IC~IE=%.2fmA."%IC);

#Applying Kirchhoff's law along the output circuit
VCE=VCC-(IC/1000)*(RC+RE);                 #Collector-emitter voltage, V

#Results
print("VCE=%.2fV, is approximately half of VCC=%dV \n therefore it is mid-point biased."%(VCE,VCC));

from math import sqrt

#Variable declaration
VCC=10.0;                    #Collector supply voltage, V
IC=10.0;                     #Collector  current, mA   
VBE=0.7;                     #Base-emitter voltage, V
R1=1.5;                      #Resistor R1, kΩ 
R2=680.0;                    #Resistor R2, Ω  
RC=260.0;                    #Collector resistor, Ω 
RE=240.0;                    #Emitter resistor, Ω 
beta_min=100;                #Minimum value of base current amplification factor
beta_max=400;                #Maximum value of base current amplification factor

#Calculations
#Voltage divder rule across R1 and R2
V2=round((VCC*R2/1000)/(R1+R2/1000),2);               #Voltage drop across resistor R2, V
IE=round((V2-VBE)/(RE/1000),0);                         #OHM' LAW, Emitter current, mA
IC=IE;                                  #Collector current(approx. equal to emitter current),mA
beta_avg=sqrt(beta_min*beta_max);       #Average value of base current amplification factor
IB=IE/(beta_avg +1);                    #Base current, mA
IB=IB*1000;                             #Base current, 𝜇A

#Results
print("Base current= %.2f 𝜇A"%IB);

#Variable declaration
VEE=12.0;                    #Emitter supply voltage, V
RC=1.5;                     #Collector resistor, kΩ
RB=120.0;                   #Base resistor kΩ
RE=510.0;                   #Emitter resistor, Ω                 
VBE=0.7;                    #Base-emitter voltage, V
beta=60.0;                  #Base current amplification factor

#Calculations
#Applying Kirchhoff's voltage law,
#IB*RB - VBE - IE*RE +VEE=0
#Since IE~IC and IC=beta*IB,
IB=(VEE-VBE)/(RB + beta*RE/1000);           #Base current , mA
IC=round(beta*IB,1);                                 #Collector current, mA

#Applying Kirchhoff's voltage law along output circuit,
VCE=VEE-IC*(RC + RE/1000);                        #Collector-emitter voltage, V


#Results
print("Operating point : VCE= %.2fV and IC=%.1fmA."%(VCE,IC));

from math import floor

#Variable declaration
VEE=9.0;                    #Emitter supply voltage, V
RC=1.2;                     #Collector resistor, kΩ
RB=100.0;                   #Base resistor ,kΩ
VBE=0.7;                    #Base-emitter voltage, V
beta=45.0;                  #Base current amplification factor

#Calculations
#Applying Kirchhoff's voltage law,
#IB*RB + VBE=VEE
#Since IE~IC and IC=beta*IB,
IB=round((VEE-VBE)/RB,3);           #Base current , mA
IC=floor(beta*IB*100)/100;                                 #Collector current, mA

#Applying Kirchhoff's voltage law along output circuit,
VCE=VEE-IC*RC;                        #Collector-emitter voltage, V

#Results
print("Operating point : VCE= %.2fV and IC=%.2fmA."%(VCE,IC));

#Variable declaration
VCC=16.0;                   #Collector supply voltage, V
VBE=0.7;                    #Base-emitter voltage, V
IC=1.0;                     #Collector current, mA
VCE=6.0;                    #Collector-emitter voltage, V
beta=150.0;                 #Base current amplification factor

#Calculations
#For a good design, VE=VCC/10;
VE=VCC/10;                  #Emitter terminal's voltage, V
#OHM's Law
#And, taking IE~IC
RE=VE/IC;                   #Emitter resistor, kΩ

#Applying Kirchhoff's voltage law alog output circuit:
#VCC=IC*RC + VCE + VE
RC=(VCC-VCE-VE)/IC;                       #Collector resistor, kΩ
V2=VE+VBE;                                #Voltage drop across resistor R2,V
#From the relation I1=10*IB
R2=(beta*RE)/10;                          #Resistor R2, kilo ohm

#From voltage divider rule across R1 and R2,
#V2=(VCC*R2)/(R1+R2)
R1=(VCC-V2)*R2/V2;                          #Resistor R1, kΩ            

#Results
print("RE=%.1f kΩ , RC=%.1f kΩ, R1=%.0f kΩ and R2=%d kΩ."%(RE,RC,R1,R2));

#Variable declaration
ICBO=5.0;                          #Collector to base leakage current, microampere
beta=40.0;                         #Base current amplification factor
IC_zero_signal=2.0;                #Zero signal collector current, mA
op_temp=25.0;                      #operating temperature, degree celsius
temp_risen=55.0;                   #Temperature risen, degree celsius
temp_ICBO_doubles=10.0;            #Temperature after which ICBO doubles, degree celsius

#Calculations

#(i)
ICEO=(beta+1)*ICBO;                 #Collector to emitter leakage current, microampere

#(ii)
Number_of_times_ICBO_doubled=(temp_risen - op_temp)/temp_ICBO_doubles;  #Number of times ICBO doubles
ICBO_final=ICBO*2**Number_of_times_ICBO_doubled;                        #Final value of collector to base leakage current, microampere
ICEO_final=ICBO_final*(beta + 1);                                       #Final value of collector to emitter leakage current, microampere

IC_zero_signal_55=(ICEO_final/1000) +IC_zero_signal;                    #Zero signal collector current at 55 degree celius
change=(IC_zero_signal_55-IC_zero_signal)*100/IC_zero_signal;           #Percentage change in zero signal collector current

#Result
print("(i) The percentage change in the zero signal collector current=%.0f%%. "%change)

#(iii)
#For the silicon transistor
ICBO=0.1;                          #Collector to base leakage current, microampere

ICEO=(beta+1)*ICBO;                 #Collector to emitter leakage current, microampere

Number_of_times_ICBO_doubled=(temp_risen - op_temp)/temp_ICBO_doubles;  #Number of times ICBO doubles
ICBO_final=ICBO*2**Number_of_times_ICBO_doubled;                        #Final value of collector to base leakage current, microampere
ICEO_final=ICBO_final*(beta + 1);                                       #Final value of collector to emitter leakage current, microampere

IC_zero_signal_55=(ICEO_final/1000) +IC_zero_signal;                    #Zero signal collector current at 55 degree celius
change=(IC_zero_signal_55-IC_zero_signal)*100/IC_zero_signal;           #Percentage change in zero signal collector current


#Result
print("(ii) The percentage change in the zero signal collector current=%.1f%%. "%change)

#Variable declaration
ICBO=0.02                          #Collector to base leakage current, 𝜇A
alpha=0.99;                        #Current amplification factor
IE=1.0;                            #Emitter current, mA
op_temp=27.0;                      #operating temperature, degree celsius
temp_risen=57.0;                   #Temperature risen, degree celsius
temp_ICBO_doubles=6.0;             #Temperature after which ICBO doubles, degree celsius

#Calculations
Number_of_times_ICBO_doubled=(temp_risen - op_temp)/temp_ICBO_doubles;       #Number of times ICBO doubles
ICBO_55=ICBO*2**Number_of_times_ICBO_doubled;                                #collector to base leakage current at 55 degree celsius, 𝜇A
IC=alpha*IE + ICBO_55/1000;                                                     #Collector current, mA
IB=IE-IC;                                                                    #Base current, mA
IB=IB*1000;                                                                  #Base current,𝜇A

#Results
print("Base current at 57 degree celsius=%.1f 𝜇A "%IB);

