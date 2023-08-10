#Variable declaration
I_DSS=12.0;                                 #Shorted gate drain current, mA
V_GS_off=-5.0;                              #Gate-source cut-off voltage, V

#Result
print("ID=%d[1 + VGS/%d]²mA."%(I_DSS,abs(V_GS_off)));

#Variable declaration
I_DSS=32.0;                                 #Shorted gate drain current, mA
V_GS_off=-8.0;                              #Gate-source cut-off voltage, V
V_GS=-4.5;                                  #Gate-source voltage, V


#Calculation
I_D=I_DSS*(1-(V_GS/V_GS_off))**2;                    #Drain current mA

#Result
print("The drain current=%.2fmA."%I_D);

from math import sqrt

#Variable declaration
I_DSS=10.0;                                 #Shorted gate drain current, mA
V_GS_off=-6.0;                              #Gate-source cut-off voltage, V
I_D=5.0;                                    #Drain current mA


#Calculation
#(i)
#Since, I_D=I_DSS*[1 - (V_GS/V_GS_off)]²
V_GS=V_GS_off*(1-sqrt(I_D/I_DSS));              #Gate-source voltage, V

#(ii)
V_P=-V_GS_off;                                  #Pinch-off voltage, V        

#Result
print("(i) VGS=%.2fV."%V_GS);
print("(ii) VP=%dV"%V_P);

#Variable declaration
V_GS_off=-4.0;                              #Gate-source cut-off voltage, V
I_DSS=12.0;                                 #Shorted gate drain current, mA
R_D=560.0;                                  #Drain resistor, Ω

#Calculation
V_P=-V_GS_off;          #Pinch-off voltage, V
V_DS=V_P;               #Minimum drain-source voltage for JFET to be in constant current region, V
I_D=I_DSS;              #Maximum drain current, mA (V_GS=0)
V_RD=(I_D/1000)*R_D;    #Voltage across drain resistor, V (OHM's LAW)
V_DD=V_DS+V_RD;         #Minimum value of supply voltage to drain, V

#Result
print("The minimum value of VDD required =%.2fV."%V_DD);

#Variable declaration
I_DSS=3.0;                                 #Shorted gate drain current, mA
V_GS_off=-6.0;                              #Gate-source cut-off voltage, V
V_GS=-2.0;                                  #Gate-source voltage, V


#Calculation
I_D=I_DSS*(1-(V_GS/V_GS_off))**2;                    #Drain current mA

#Result
print("The drain current=%.2fmA."%I_D);

#Variable declaration
VGS_off=4;                #Gate-source cut-off voltage, V
VGS=6;                    #Gate source voltage, V

print("p-channel JFET requires a positive gate-to-source voltage to pass drain current.");
print("More positive voltage, the less the drain current. ");
print("Any further increase in VGS keeps the JFET cut-off. Therefore, ID=0A.");

#Variable declaration
V_GS=15.0;              #Gate-source voltage, V
I_G=1e-03;               #Gate current, μA

#Calculation
R_GS=(V_GS/(I_G*10**-6))/10**6;          #Gate to source resistance, MΩ (OHM's LAW)

#Result
print("The gate to source resistance=%dMΩ."%R_GS);


#Variable declaration
V_GS_max=-3.1;                      #Maximum gate to source voltage, V
V_GS_min=-3.0;                      #Minimum gate to source voltage, V
I_D_max=1.3;                      #Maximum drain current, mA
I_D_min=1.0;                      #Minimum drain current, mA


#Calculation
delta_V_GS=abs(V_GS_max-V_GS_min);              #Change in gate to source voltage, V
delta_I_D=I_D_max-I_D_min;                      #Change in drain current, mA
g_fs=(delta_I_D/delta_V_GS)*1000;               #Transconductance, μ mho


#Result
print("Transconductance=%.0f μ mho"%g_fs);

#Variable declaration
V_GS=[0,0,-0.2];            #Readings of Gate-source voltage, V
V_DS=[7,15,15];             #Readings of Drain-source voltage, V
ID=[10,10.25,9.65];         #Readings of drain current, mA


#Displaying the readings:
print("VGS= %dV     %dV        %.1fV"%(V_GS[0],V_GS[1],V_GS[2]));
print("VDS= %dV     %dV        %dV"%(V_DS[0],V_DS[1],V_DS[2]));
print("ID = %dV    %.2fV     %.2fV"%(ID[0],ID[1],ID[2]));

#Calculations
#(i)
#V_GS constant at 0V,
delta_VDS=V_DS[1]-V_DS[0];          #Change in drain-source voltage, V
delta_ID=ID[1]-ID[0];               #Change in drain current, mA
rd=delta_VDS/delta_ID;              #a.c drain resistance, kΩ

#(ii)
#V_DS constant at 15V,
delta_VGS=V_GS[2]-V_GS[1];                  #Change in gate-source voltage, V
delta_ID=ID[2]-ID[1];                       #Change in drain current, mA
g_fs=round((delta_ID/delta_VGS)*1000,);     #Transconductance, μ mho

#(iii)
amplification_factor=rd*1000*g_fs*10**-6;                          #Amplification factor


#Result
print("(i)   The a.c drain resistance=%dkΩ."%rd);
print("(ii)  The transconductance=%d μ mho."%g_fs);
print("(iii) The amplification factor=%d."%amplification_factor );

#Variable declaration
g_mo=4000.0;                        #Maximum transconductance, μS
V_GS=-3.0;                          #Gate to source voltage, V
V_GS_off=-8.0;                      #Gate-source cut-off voltage, V

#Calculation
g_m=g_mo*(1-(V_GS/V_GS_off));               #Transconductance, μS

#Result
print("The transconductance=%d μS."%g_m);

#Variable declaration
g_mo=5000.0;                        #Maximum transconductance, μS
V_GS=-4.0;                          #Gate to source voltage, V
V_GS_off=-6.0;                      #Gate-source cut-off voltage, V
I_DSS=3.0;                          #Shorted-gate drain current, mA

#Calculation
g_m=g_mo*(1-(V_GS/V_GS_off));               #Transconductance, μS
I_D=(I_DSS*(1-(V_GS/V_GS_off))**2)*1000;                    #Drain current μA


#Result
print("The transconductance=%.0f μS."%g_m);
print("The drain current=%d μA."%I_D);

#Variable declaration
V_GS_off=-8.0;                      #Gate-source cut-off voltage, V
I_DSS=16.0;                         #Shorted-gate drain current, mA
R_D=2.2;                            #Drain resistor, kΩ
R_G=1.0;                            #Gate resistor, MΩ
V_DD=10.0;                          #Drain supply voltage, V
V_GG=-5.0;                           #Gate supply voltage, V


#Calculation
V_GS=V_GG;                                           #Gate-source voltage, V
I_D=I_DSS*(1-(V_GS/V_GS_off))**2;                    #Drain current μA
V_DS=V_DD-I_D*R_D;                                   #Drain-source voltage, V (Kirchhoff's voltage law)

#Result
print("The gate-source voltage=%dV."%V_GS);
print("The drain current=%.2fmA."%I_D);
print("The drain-source voltage=%.2fV."%V_DS);

#Variable declaration
I_D=5.0;                   #Drain current mA
V_DD=15.0;                 #Drain supply voltage, V
V_G=0;                     #Gate voltage, V
R_D=1.0;                   #Drain resistor, kΩ
R_S=470.0;                 #Source resistor, Ω


#Calculation
V_S=(I_D/1000)*R_S;                 #Source voltage, V (OHM's LAW)
V_D=V_DD-I_D*R_D;                   #Drain voltage, V (Kirchhoff's voltage law)
V_DS=V_D-V_S;                       #Drain-source voltage, V
V_GS=V_G-V_S;                       #Gate-source voltage, V

#Result
print("The drain-source voltage=%.2fV."%V_DS);
print("The gate-source voltage=%.2fV."%V_GS);

#Variable declaration
V_GS=-5.0;                     #Gate-source voltage, V
I_D=6.25;                      #Drain current mA


#Calculation
R_S=abs(V_GS/(I_D/1000));              #Required source resistor, Ω (OHM's LAW)

#Result
print("The required source resistor=%d Ω."%R_S);

#Variable declaration
I_DSS=25.0;                                 #Shorted gate drain current, mA
V_GS_off=15.0;                              #Gate-source cut-off voltage, V
V_GS=5.0;                                   #Gate-source voltage, V


#Calculation
I_D=I_DSS*(1-(V_GS/V_GS_off))**2;               #Drain current mA
R_S=V_GS/(I_D/1000);                            #Required source resistor, Ω (OHM's LAW)


#Result
print("The source resistance=%.0fΩ."%R_S);

#Variable declaration
I_DSS=15.0;                                 #Shorted gate drain current, mA
V_GS_off=-8.0;                              #Gate-source cut-off voltage, V
V_DD=12.0;                                  #Drain supply voltage,V
V_D=V_DD/2;                                 #Drain voltage(half of V_DD), V

#Calculation
I_D=I_DSS/2;                                #Drain current(approximately half of I_DSS), mA
V_GS=V_GS_off/3.4;                          #Gate-source voltage, V
R_S=abs(V_GS/(I_D/1000));                   #Source resistor, Ω (OHM's LAW)
#Since,V_D=V_DD-I_D*R_D;                      
R_D=(V_DD-V_D)/(I_D/1000);                         #Drain resistor, Ω (OHM's LAW)

#Result
print(" RS=%d Ω and RD=%d Ω."%(R_S,R_D));

from math import sqrt

#Variable declaration
I_DSS=5.0;                       #Shorted gate drain current, mA
V_GS_off=-2.0;                   #Gate-source cut-off voltage, V
V_DS=10.0;                       #Drain-source voltage,V
I_D=1.5;                         #Drain current, mA
V_DD=20.0;                       #Drain supply voltage,V
V_G=0;                           #Gate voltage, V


#Calculation
#Since,  Drain current, I_D=I_DSS*(1-(V_GS/V_GS_off))**2;                    
V_GS=V_GS_off*(1-sqrt(I_D/I_DSS));              #Gate-source voltage, V

#Since, V_GS=V_G-V_S,
V_S=V_G-V_GS;                           #Source voltage, V

R_S=V_S/I_D;                            #Source resistor, kΩ

#Since, V_DD=I_D*R_D +V_DS+ I_D*R_S,
R_D=(V_DD-I_D*R_S-V_DS)/I_D;                    #Drain resistor, kΩ

#Calculation
print("The source resistance=%.1f kΩ"%R_S);
print("The drain resistance=%d kΩ."%R_D);

#Variable declaration
V_DD=30.0;                  #Drain supply voltage, V
R_D=5.0;                    #Drain resistor, kΩ
I_D=2.5;                    #Drain current, mA
R_S=200.0;                  #Source resistor, Ω

#Calculation
#(i)
V_DS=V_DD-I_D*(R_D+(R_S/1000));                 #Drain-source voltage, V

#(ii)
V_GS=-(I_D/1000)*R_S;                          #Gate-source voltage, V

#Result
print("The drain-source voltage=%dV."%V_DS);
print("The gate-source voltage=%.1fV."%V_GS);

#Variable declaration
ID_1=2.15;                  #First stage drain current, mA
ID_2=9.15;                  #Second stage drain current, mA
VDD=30;                     #Drain supply voltage, V
RS_1=0.68;                  #Source resistance of 1st stage, kΩ
RS_2=0.22;                  #Source resistance of 2nd stage, kΩ
RD_1=8.2;                   #Drain resistor of 1st stage, kΩ
RD_2=2;                     #Drain resistor of 2nd stage, kΩ

#Calculation
V_RD1=ID_1*RD_1;                #Voltage drop across 8.2kΩ
VD_1=VDD-V_RD1;                 #Drain voltage of 1st stage, V
VS_1=ID_1*RS_1;                 #D.C potential of source of first stage, V
V_RD2=ID_2*RD_2;                #Voltage drop across 2kΩ
VD_2=VDD-V_RD2;                 #Drain voltage of 2nd stage, V
VS_2=ID_2*RS_2;                 #D.C potential of source of 2nd stage, V


#Result
print("drain voltage of 1st stage=%.2fV."%VD_1);
print("Source voltage of 1st stage=%.2fV."%VS_1);
print("drain voltage of 2nd stage=%.1fV."%VD_2);
print("Source voltage of 2nd stage=%.2fV."%VS_2);

#Variable declaration
VDD=12;                     #Drain supply voltage, V
VD=7;                       #Drain voltage, V
R1=6.8;                     #Resistor R1, MΩ
R2=1;                       #Resistor R2, MΩ
RS=1.8;                     #Source resistance, kΩ
RD=3.3;                     #Drain resistor, kΩ

#Calculation
ID=(VDD-VD)/RD;                  #Second stage drain current, mA
VS=ID*RS;                        #Source voltage, V
VG=VDD*R2/(R1+R2);               #Drain voltage, V
VGS=VG-VS;                       #Drain-source voltage, V

#Calculation
print("Drain current=%.2fmA."%ID);
print("Gate-source voltage=%.1fV."%VGS);

from math import sqrt

#Variable declaration
VDD=30;                     #Drain supply voltage, V
ID=2.5;                     #Drain current, mA
VDS=8;                      #Drain-source voltage, V
VGS_off=-5;                 #Gate-source cutoff voltage, V
R1=1;                       #Resistor R1, MΩ
R2=500;                     #Resistor R2, kΩ
IDSS=10;                    #Shorted gate drain current, mA

#Calculation
#ID=IDSS*square_of(1-(VGS/VGS_off))
VGS=VGS_off*(1-sqrt(ID/IDSS));           #Gate-source voltage, V
V2=VDD*R2/(R1*1000+R2);                       #Voltage across R2, V


#V2=VGS+ID*RS
RS=(V2-VGS)/ID;                         #Source resistor, kΩ

#Result
print("Source resistor, RS=%dkΩ."%RS);

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#Variable declaration
VDD=20.0;                 #Drain supply voltage, V
RS=50.0;                  #Source resistor, Ω
RD=150.0;                 #Drain resistor, Ω

#Calculation
VDS_max=VDD;                           #Maximum drain source voltage, V
ID_max=(VDD/(RD+RS))*1000;             #Maximum drain current, mA


#plot
x=[i for i in range(0,(int)(VDS_max+1))];          #Plot variable for V_DS
y=[(i/(RD+RS))*1000 for i in reversed(x[:])];      #Plot variable for ID


plt.plot(x,y);
plt.xlabel("VDS(V)");
plt.ylabel("ID(mA)");
plt.title("d.c load line");

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#Variable declaration
VDD=20;                 #Drain supply voltage, V
RD=500.0;                 #Drain resistor, Ω

#Calculation
VDS_max=VDD;                           #Maximum drain source voltage, v 
ID_max=(VDD/RD)*1000;                  #Maximum drain current, mA

#Plot
x=[i for i in range(0,(int)(VDS_max+1))];          #Plot variable for V_DS
y=[(i/RD)*1000 for i in reversed(x[:])];      #Plot variable for ID

plt.plot(x,y);
plt.xlabel("VDS(V)");
plt.ylabel("ID(mA)");
plt.title("d.c load line");

#Variable declaration
VDD=20;                 #Drain supply voltage, V
RD=12.0;                  #Drain resistor, kΩ
RL=8.0;                   #Load resistor, kΩ
RG=1.0;                   #Gate resistor, MΩ
gm=1.0;                   #transconductance, mA/V

#Calculation
gm=gm*10**-3;                   #transconductance, mho
RAC=(RD*RL)/(RD+RL);            #Total a.c load, kΩ
Av=gm*RAC*1000;                 #Voltage gain


#Result
print("Voltage gain=%.1f."%Av);

#Variable declaration
gm=3000;                   #transconductance, μmho
RD=10;                     #Drain resistance, kΩ

#Calculation
Av=gm*10**-6*RD*1000;       #Voltage gain

#Result
print("Voltage gain=%d."%Av);

#Variable declaration
IDSS=8;                     #Shorted gate drain current, mA
VGS_off=-10;                #Gate-source cut-off voltage, V
ID=1.9;                     #Drain current, mA
RD=3.3;                     #Drain resistance, kΩ
RS=2.7;                     #Source resistor, kΩ
vin=100;                    #Input voltage, mV

#Calculation
VGS=-ID*RS;                      #Gate-source voltage, V
gmo=2*IDSS*10**-3/abs(VGS_off);  #Maximum transconductance, S
gm=gmo*(1-(VGS/VGS_off));        #Transconductance, S
Av=gm*RD*1000;                   #Voltage gain
vout=Av*vin;                     #Output voltage, mA

#Result
print("Output voltage=%dmV(r.m.s)."%vout);
                

#Variable declaration
RL=4.7;                    #Load resistor, Ω
RD=3.3;                     #Drain resistance, kΩ
gm=779*10**-6;              #Transconductance, S
vin=100;                    #Input voltage, mV


#Calculation
RAC=RD*RL/(RD+RL);          #Total a.c drain resistance, kΩ
Av=gm*RAC*1000;             #Voltage gain
vout=Av*vin;                     #Output voltage, mA

#Result
print("Output voltage=%dmV(r.m.s)."%vout);

#Variable declaration
RD=1.5;            #Drain resistance, kΩ
gm=4;              #Transconductance, mS
RS=560;            #Source resistance, Ω

#Calculation
Av=gm*10**-3*RD*1000/(1+gm*10**-3*RS);
print("Voltage gain=%.2f."%Av);

#If RS is bypassed by a capacitor
Av=gm*10**-3*RD*1000;
print("Voltage gain, if RS resistor is bypassed=%d."%Av);

#Variable declaration
from math import sqrt

IDSS=10;                     #Shorted gate drain current, mA
VGS_off=-3.5;                #Gate-source cut-off voltage, V
RD=1.5;            #Drain resistance, kΩ
RS=750;            #Source resistance, Ω


#Calculation
#From d.c biasing
ID=2.3;                                     #Drain current, mA
VGS=round(VGS_off*(1-sqrt(ID/IDSS)),1);              #Gate-source voltage, V
gm=round(round((2*IDSS/abs(VGS_off)),1)*round((1-(VGS/VGS_off)),3),2);           #Transconductance, mS


#(i)
Av=gm*RD;                         #Voltage gain with RS resistor bypassed
print("(i) Voltage gain with RS bypassed=%.3f."%Av);

#(ii)
Av=Av/(1+gm*(RS/1000.0));
print("(ii) Voltage gain with RS unbypassed=%.2f."%Av);

#Variable declaration
IDSS=10.0;                     #Shorted gate drain current, mA
VGS_off=-8.0;                  #Gate-source cut-off voltage, V

#Calculation
#(i)
if(VGS_off<0):
    print("(i)  n-channel D-MOSFET");
else:
    print("(i)  p-channel D-MOSFET");
    

#(ii)
VGS=-3.0;                                #Gate-source voltage, V
ID=IDSS*(1-(VGS/VGS_off))**2;          #Drain current mA
print("(ii) Drain current=%.2fmA"%ID);

#(iii)
VGS=3.0;                                #Gate-source voltage, V
ID=IDSS*(1-(VGS/VGS_off))**2;          #Drain current mA
print("(iii) Drain current=%.1fmA"%ID);

#Variable declaration
IDSS=1.0;                     #Shorted gate drain current, mA
VGS_off=-6.0;                  #Gate-source cut-off voltage, V


#Calculation
#Point 1
VGS=0;                  #Gate source voltage, V               
ID=IDSS;                #Drain current, mA
print("Point 1: VGS=%dV and ID=%dmA."%(VGS,ID));

#Point 2
VGS=VGS_off;                  #Gate source voltage, V               
ID=0;                        #Drain current, mA
print("Point 2: VGS=%dV and ID=%dmA."%(VGS,ID));

#locating more points by changing VG values
VGS=-3;                  #Gate source voltage, V               
ID=IDSS*(1-(VGS/VGS_off))**2;          #Drain current mA
print("Point 3: VGS=%dV and ID=%.2fmA."%(VGS,ID));

VGS=-1;                  #Gate source voltage, V               
ID=IDSS*(1-(VGS/VGS_off))**2;          #Drain current mA
print("Point 4: VGS=%dV and ID=%.3fmA."%(VGS,ID));

VGS=1;                  #Gate source voltage, V               
ID=IDSS*(1-(VGS/VGS_off))**2;          #Drain current mA
print("Point 5: VGS=%dV and ID=%.2fmA."%(VGS,ID));

VGS=3;                  #Gate source voltage, V               
ID=IDSS*(1-(VGS/VGS_off))**2;          #Drain current mA
print("Point 6: VGS=%dV and ID=%.2fmA."%(VGS,ID));

#Variable declaration
VDD=18;          #Drain supply voltage, V
RD=620.0;           #Drain resistor, Ω
IDSS=12.0;        #Shorted gate drain current, mA
VGS_off=-8.0;      #Gate-source cut-off voltage, V


#Calculation
ID=IDSS;               #Drain current, mA
VDS=VDD-IDSS*(RD/1000);       #Drain source voltage, V

#Result
print("Drain source voltage=%.1fV."%VDS);

#Variable declaration
VDD=15;           #Drain supply voltage
RD=620.0;          #Drain resistor, Ω
RL=8.2;          #Load resistor, kΩ
vin=500.0;         #Input voltage, V
IDSS=12.0;        #Shorted gate drain current, mA
gm=3.2;            #Transconductance, mS

#Calculation
#(i)
VDS=VDD-IDSS*(RD/1000.0);       #Drain source voltage, V

#(ii)
RAC=RD*RL*1000/(RD+RL*1000);  #Total a.c drain resistace, Ω
vout=(gm/1000.0)*RAC*vin;     #Output voltage, V

#Result
print("(i)  Drain source voltage=%.2fV."%VDS);
print("(ii) Output voltage=%dmV"%vout);

#Variable declaration
ID_on=500.0;             #Drain current for MOSFET ON, mA
VGS_on=10.0;             #Gate-source voltage for MOSFET ON, V
VGS_th=1.0;              #Threshold value of gate-source voltage, V
VGS=5;                 #Gate-source voltage, V

#Calculation
K=round(ID_on/(VGS_on-VGS_th)**2,2);   #Constant for a E-MOSFET, mA/V²
ID=K*(VGS-VGS_th)**2;                  #Drain current, mA

#Result
print("Drain current=%.1fmA"%ID);

#Variable declaration
ID_on=3.0;             #Drain current for MOSFET ON, mA
VGS_on=10.0;             #Gate-source voltage for MOSFET ON, V
VGS_th=3.0;              #Threshold value of gate-source voltage, V

#Calculation
K=round((ID_on/(VGS_on-VGS_th)**2),3);   #Constant for a E-MOSFET, mA/V²
print("K=%.3fe-03A/V²."%K);

#Determining different points for plotting
VGS=5;                                 #Gate-source voltage, V
ID=K*(VGS-VGS_th)**2;                  #Drain current, mA
print("For VGS=5V, Drain current=%.3fmA"%ID);
VGS=8;                                 #Gate-source voltage, V
ID=K*(VGS-VGS_th)**2;                  #Drain current, mA
print("For VGS=8V, Drain current=%.3fmA"%ID);
VGS=10;                                 #Gate-source voltage, V
ID=K*(VGS-VGS_th)**2;                  #Drain current, mA
print("For VGS=10V, Drain current=%.dmA"%ID);
VGS=12;                                 #Gate-source voltage, V
ID=K*(VGS-VGS_th)**2;                  #Drain current, mA
print("For VGS=12V, Drain current=%.2fmA"%ID);

#Variable declaration
VDD=24.0;             #Drain supply voltage, V
RD=470.0;             #Drain resistor, Ω
R1=100.0;             #Resistor R1, kΩ
R2=15.0;              #Resistor R2, kΩ
ID_on=500.0;             #Drain current for MOSFET ON, mA
VGS_on=10.0;             #Gate-source voltage for MOSFET ON, V
VGS_th=1.0;              #Threshold value of gate-source voltage, V


#Calculation
VGS=VDD*R2/(R1+R2);                      #Gate-source voltage, V (Voltage divider rule)
K=round((ID_on/(VGS_on-VGS_th)**2),2);   #Constant for a E-MOSFET, mA/V²
ID=K*(VGS-VGS_th)**2;                    #Drain current, mA
VDS=VDD-(ID/1000)*RD;                           #Drain-source voltage, V

#Result
print("Drain-source voltage=%.1fV."%VDS);

#Variable declaration
VDD=20.0;             #Drain supply voltage, V
RD=1.0;               #Drain resistor, kΩ
RG=5.0;               #Gate resistor , MΩ
ID_on=10.0;             #Drain current for MOSFET ON, mA

#Calculation
#since, VGS=VDS
ID=ID_on;              #Drain current, mA
VDS=VDD-ID*RD;         #Drain-source voltage, V

#Result
print("Drain current=%dmA."%ID);
print("Drain-source voltage=%dV."%VDS);

#Variable declaration
VDD=10.0;             #Drain supply voltage, V
RD=3.0;             #Drain resistor, kΩ
R1=1.0;             #Resistor R1, MΩ
R2=1.0;              #Resistor R2, MΩ
ID_on=10.0;             #Drain current for MOSFET ON, mA
VGS_on=10.0;             #Gate-source voltage for MOSFET ON, V
VGS_th=1.5;              #Threshold value of gate-source voltage, V

#Calculation
K=round((ID_on/(VGS_on-VGS_th)**2),3);   #Constant for a E-MOSFET, mA/V²
VGS=VDD*R2/(R1+R2);                      #Gate-source voltage, V (Voltage divider rule)
ID=K*(VGS-VGS_th)**2;                    #Drain current, mA

#Result
print("Drain current=%.2fmA."%ID);



