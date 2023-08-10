import tensorflow as tf
import numpy as np
# Build tensorflow graph of single simulation step return the new concentration x after simulation step
def SimStep(x,In_Yr,In_Yp,In_k,In_dt): 
    Yr=tf.constant(In_Yr,tf.float32)# Reactant ratios for each reaction
    Yp=tf.constant(In_Yp,tf.float32)# Product ratios for each reaction
    k=tf.constant(In_k,tf.float32)# Reaction constants for each reaction
    dt=tf.constant(In_dt,tf.float32) # time lapse for each simulation step
    s1=tf.pow(x,Yr)
    s2=tf.reduce_prod(s1,1)
    r=k*s2#Reacion rates 
    s4=tf.scalar_mul(dt,r)
    Yd=Yp-Yr # Change in concentrations attribute to each reaction
    dxij=s4*tf.transpose(Yd)# concentration changes each reaction in this step***
    dx=tf.reduce_sum(dxij,1) #sum of concentration changes from all reactions in this step***
    xp=x+dx#New concentration after steps  
    return(xp)

x0=[0.4,0.3,0,0]# intial concentrations for reaction components. Initiale concentration of a and b will be 0.4,0.3
x_names=['a','b','c','d']# names of components in the reaction associate with conecntration stored in x
In_Yr=[[1,2,0,0],[0,0,1,0]]# Set reactant ratios for each reaction
In_Yp=[[0,0,1,0],[0,0,0,1]]# Set product ratios for each reaction
In_k=[8,4]# Set constant for each reaction
In_dt=0.01# Set time lapse for each simulation step
x=tf.placeholder(tf.float32) # conentrations for each component this is placeholder and not variable/constant since we need to update it each cycle
xp=SimStep(x,In_Yr,In_Yp,In_k,In_dt) # Build tensorflow graph of single simulation step return the new concentration x after simulation step

sess = tf.InteractiveSession() # start interactive session  the difference betwee this and regular session is that interactive session established itself as the defult session hence you dont need to specify the session every time you use eval
sess.run(tf.initialize_all_variables())# initialize variables
NumSteps=1000# number of simulation step
Lapse=10# lapses between which data will be gather and saved for graph
Tlapse=np.zeros(round(NumSteps/Lapse)+1,dtype=np.float32)# array containing the times in which the data were collected
x_sample=np.zeros([round(NumSteps/Lapse)+1,len(x_names)],dtype=np.float32)# sample x (concentrations)  during simulation 
Tlapse[0]=0 # Set initial simulation time as zero
x_sample[0]=x0 # record initial commponent concentration

for i in range(NumSteps):# Run simulation for NumSteps
    x0=sess.run(xp,feed_dict={x:x0})# run numeric simulation step
    if (i%Lapse)==0:# collect the concentration in lapse of several simulation steps for graph (once every Lapse Steps)
        x_sample[round(i/Lapse)+1]=x0
        Tlapse[round(i/Lapse)+1]=i*In_dt
        

import matplotlib.pyplot as plt
for c,label in zip(x_sample.swapaxes(0,1),x_names):
    plt.plot(Tlapse,c,label=label)
plt.legend(loc='upper left')
plt.xlabel("Time")
plt.ylabel("concentration")
plt.show()
plt.savefig("Concentration.png")# Save graph as png image file







