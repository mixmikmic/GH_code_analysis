import numpy as np
import random as random
import matplotlib.pyplot as plt
import time
import pandas as pd
import math
from scipy.spatial import distance#
from pyproj import Proj,transform
import seaborn as sns
import copy
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
v84 = Proj(proj="latlong",towgs84="0,0,0",ellps="WGS84")
v36 = Proj(proj="latlong", k=0.9996012717, ellps="airy", towgs84="446.448,-125.157,542.060,0.1502,0.2470,0.8421,-20.4894")
vgrid = Proj(init="world:bng")
#Calculate Euclidean distances using H

# def dfLLtoEN(df):
#     """Returns easting, northing tuple
#     """
#     vlon36, vlat36 = transform(v84,v36,df["long"].values,df["lat"].values)
#     result = vgrid(vlon36,vlat36)

#     # Transform the output to a Dataframe
#     eastnorth = pd.DataFrame(index=df.index)
#     for i in result:
#         eastnorth["Easting"] = result[0]
#         eastnorth["Northing"] = result[1]

#     return round(eastnorth)

# Path = '../StudyGroup/'
# CompleteData=pd.read_csv(Path+"completeData2.csv",header = 0)
# CompleteData = CompleteData.drop(columns = 'Unnamed: 0')

# CompleteData[['Easting', 'Northing']] = dfLLtoEN(CompleteData[['lat', 'long']])

# CompleteData = CompleteData.rename(index=str, columns={'X__1': "study"})

# CompleteData.columns.values

# ImputedData = pd.read_csv('imputed_farm_reducedsize', header = 0, sep = '\t')
# ImputedData = ImputedData.rename(index=str, columns={'Unnamed: 0': "study"})

# ImputedData[['Easting', 'Northing']] = dfLLtoEN(ImputedData[['lat', 'long']])

# ImputedData['sr'] = ImputedData['ruminant']

# Data = pd.concat([CompleteData, ImputedData], ignore_index=True)

# xmax = np.max(Data['Easting'].values)
# xmin = np.min(Data['Easting'].values)
# ymax = np.max(Data['Northing'].values)
# ymin = np.min(Data['Northing'].values)
# sizex = xmax-xmin
# sizey = ymax-ymin
# Data['xcoord'] = Data['Easting'] - xmin
# Data['ycoord'] = Data['Northing'] - ymin

# Data.to_csv('All_data', sep='\t')

Data=pd.read_csv("All_data",sep = '\t', header = 0)

Copy1 = pd.read_csv('All_data', sep = '\t', header = 0)

copy2 = Copy1.values
true_cattle =  copy.deepcopy(copy2[:,3])
true_sheep =  copy.deepcopy(copy2[:,13])
cattle = copy2[:,3]
sheep = copy2[:,13]

xcoord = Data['xcoord'].values
ycoord = Data['ycoord'].values

joinedinput = np.column_stack((xcoord, ycoord))

N = len(Data)
N

dist = distance.cdist(joinedinput, joinedinput, 'euclidean')
dist = dist/100000

equipment_list = Data['equipment'].values
shares_water_list = Data['water'].values
shares_grazing_list = Data['grazing'].values
shares_milk_list = Data['milk'].values
shares_vet_list = Data['vet'].values
contact_animal_list = Data['contact_animal'].values
contact_human_list = Data['contact_people'].values

# tr = np.random.negative_binomial(1, 1/2, N) #Draw immune periods (1 day)
tr = 2*np.ones(N)
print(np.mean(tr))
        
psi = 0.00657
# psi = 2.5
nu = 1.99*(10**(-4.8))
xi = 4.65
zeta = 2.80
chi = 0.403
phi = 0.799
rho = 0.000863


epsilon = np.zeros(N)
s = np.random.negative_binomial(50, 50/55, N) #Draw latent periods (5 days)
r = np.random.negative_binomial(30, 30/38, N) #Draw infectious periods (8 days)
# r = np.random.negative_binomial(8, 8/16, N)
sum(r==0)

np.max(dist)

# kerneldist =(10e4*psi)/((10e2*psi)**2 + dist**2)   # This is without a cap
kerneldist = (psi)/(psi**2 + dist**2)
# kerneldist = np.zeros(shape=(N,N))
# for i in range(len(dist)):
#     for j in range(len(dist)):
#         if dist[i,j] <= 60:
#             kerneldist[i,j] = (psi)/((psi)**2 + dist[i,j]**2)

# This takes quite a while to run, sit back and have a cuppa
# once this has run, you won't need to run it again unless your kernel restarts

shares_equipment = np.zeros(shape = (N,N)) #cap at 10km
shares_water = np.zeros(shape = (N,N)) #cap at 10km
shares_grazing = np.zeros(shape=(N,N)) #cap at 10km
shares_milk = np.zeros(shape = (N,N)) #cap at 10km
shares_vet = np.zeros(shape = (N,N)) #cap at 10km
contactanimal = np.zeros(shape = (N,N))
contacthuman = np.zeros(shape = (N,N)) 

for i in range(N):
    print(i)
    for j in range(N):
        if i != j:
            if dist[i,j] <= 10/100:
                if (shares_milk_list[i]==1 and shares_milk_list[j] ==1):
                    shares_milk[i,j] = 1
                if (shares_water_list[i]==1 and shares_water_list[j] ==1):
                    shares_water[i,j] =1
                if (shares_grazing_list[i] ==1 and shares_grazing_list[j] ==1):
                    shares_grazing[i,j] =1
                if (equipment_list[i]==1 and equipment_list[j] ==1):
                    shares_equipment[i,j] =1
                if (shares_vet_list[i]==1 and shares_vet_list[j] ==1):
                    shares_vet[i,j] = 1
            if (contact_animal_list[i]==1 and contact_animal_list[j] ==1):
                contactanimal[i,j] = 1
            if (contact_human_list[i]==1 and contact_human_list[j] ==1):
                contacthuman[i,j] = 1

q = np.random.negative_binomial(2*28, 2*28/(4*28), N) #Draw immune periods (2 months)
np.mean(q)
RingCull = 3 #km

# [a1, a2, a3,a4,a5,a6,a7] = [0.16964086, 0.16230848, 0.1756672,  0.06334324, 0.13775076, 0.13500596,0.15628351]
[a1, a2, a3,a4,a5,a6,a7] = (1/7)*np.ones(7)

transmission_matrix = (a1*shares_equipment+
 a2*shares_water+
 a3*shares_grazing + 
 a4*shares_milk + 
 a5*shares_vet + 
 a6*contactanimal + 
 a7*contacthuman)

farm_cost = np.zeros(N)
for i in range(0,N):
    farm_cost[i] = 150*sheep[i] + 150*cattle[i]

Data['cost'] = farm_cost

kerneldist[50,200]



Cumulative2 = []
Endemic2 = []
Maximum2 = []
Totalcost2 = []
Duration2 = []
Infection2 = []

for f in range(10):
    cumInfEFF = np.zeros(1)
    endemicEFF = np.zeros(1)
    maxEFF = np.zeros(1)
    costEFF = []
    timeEFF =  []
    infEFF = []
    cumInfArray = np.zeros(600)
    InfArray = np.zeros(600)
#     for inde,efficac in enumerate(efficacy):
    # for i in range(1):
    start_time = time.time()
#Choose initial cases
    cattle = copy.deepcopy(Data['cattle'].values).astype(int)
    sheep = copy.deepcopy(Data['sr'].values).astype(int)
    t = 0
    cost = np.zeros(N)
    RingVac = 0.5
    A = np.zeros(shape=(N,10))     
    A[:,0] = -1 
    vac = np.zeros(N)
    cumInf = np.zeros(600)
    numInf = np.zeros(600)

    initial1 = random.randint(0,N-1)
    initial2 = (initial1+1)%N
    initial3 = (initial2+1)%N

    infect_cow = np.zeros(N)
    infect_sheep = np.zeros(N)
    I = np.zeros(N)

    I[initial1] = 1
    I[initial2] = 1
    I[initial3] = 1

    A[initial1, ] = [initial1, 0, s[initial1], r[initial1], tr[initial1], q[initial1], 0,0,  0,0]
    A[initial2, ] = [initial2, 0, s[initial2], r[initial2], tr[initial2], q[initial2], 0,0, 0,0]
    A[initial3, ] = [initial3, 0, s[initial3], r[initial3], tr[initial3], q[initial1], 0,0, 0,0]

    infectind = [i for i in range(np.size(I)) if I[i]==2]
    susceptind = [i for i in range(np.size(I)) if I[i] ==0]
    exposedind = [i for i in range(np.size(I)) if I[i] == 1]
    immuneind = [i for i in range(np.size(I)) if I[i] ==4]
    vaccineind = [i for i in range(np.size(I)) if I[i]==3]

    Inf=[len(infectind)]
    Sus = [len(susceptind)]
    Exp = [len(exposedind)]
    Imm = [len(immuneind)]
    Vac = [len(vaccineind)]

    time_plot=[0]
    get_ipython().run_line_magic('matplotlib', 'notebook')
    start_time = time.time()
    fig = plt.figure(figsize = (10,4))
    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    plt.ion

    fig.show()
    fig.canvas.draw()
    Dose = []
    VAC = []
    travelled = []
    Capacity = 80
#     print(efficac)

    while t<599:
        infNum = 0
        t = t+1
        ################################## Reintroduction infection #################################
#         if t%90 == 0:
#             Suslen = len(np.where(I==0)[0])
#             newlyinfected = np.where(I==0)[0][random.randint(0,Suslen-1)]
#             I[newlyinfected] = 2
#             cattle[newlyinfected] = true_cattle[newlyinfected]
#             sheep[newlyinfected] = true_sheep[newlyinfected]
#             A[newlyinfected, ] = [newlyinfected, t, s[newlyinfected], r[newlyinfected], tr[newlyinfected],
#                                   q[newlyinfected], 0, 0,0,0]

        ################################### WITHIN FARM INFECTION ####################################
        for farm in infectind:
            Num_livestock = infect_cow[farm] + infect_sheep[farm]
            λ = 0.1
            inf = 1-((1-λ)**Num_livestock) 
            for animals in range(int(cattle[farm]+sheep[farm])):
                p_beta = np.random.uniform(0,1)
                if (p_beta<inf) and (infect_sheep[farm]+infect_cow[farm])<=Num_livestock:
                    if (random.random() < 0.5 and sheep[farm]>=1 and infect_sheep[farm]<sheep[farm]):
                        infect_sheep[farm] += 1
                    elif (cattle[farm]>=1 and infect_cow[farm]<cattle[farm]):
                        infect_cow[farm] +=1

        ################################# BETWEEN FARM INFECTION  #####################################

        ###############################################################################################
        ######################################### ATTRIBUTES ##########################################
#         ###############################################################################################
        transmission_matrix = (a1*shares_equipment+ #need to clear and update this everytime
        a2*shares_water+
        a3*shares_grazing + 
        a4*shares_milk + 
        a5*shares_vet + 
        a6*contactanimal + 
        a7*contacthuman)


        beta = np.zeros(N)
#         beta1 = nu*(cattle-infect_cow)**chi + (sheep-infect_sheep)**chi
        beta1 = nu*(xi*(cattle-infect_cow)**chi + (sheep-infect_sheep)**chi)
        beta3 = zeta*(infect_cow)**chi + infect_sheep**chi

        for i in range(N):
            transmission_matrix[:,i] *= (beta3)[i]
        for j in range(0,N):
            a = transmission_matrix[j, I ==2]
            b =  kerneldist[I ==2, j]
            beta[j] = beta1[j]*np.dot(a,b)
#             beta[j] = beta1[j]*np.matrix(transmission_matrix)[j,I==2]*np.matrix(kerneldist)[I == 2, j]

        prob_inf = (1 - np.exp(-beta))
        unif = np.random.uniform(0, 1, N)

        ### immunity ###
        for i in range(N):
            if (I[i] == 4) or (I[i] == 3):
                cattle[i] = 0
                sheep[i] = 0
                prob_inf[i] = 0
        for i in range(0,N):
            if (unif[i] <= prob_inf[i] and I[i] == 0):
                cattle[i] = true_cattle[i]
                sheep[i] = true_sheep[i]
                I[i] =  1
                A[i, ] = [i, t, s[i], r[i], tr[i], q[i], 0, 0,0,0]
        #########################################################################################################
        ########################################### NO ATTRIBUTES ###############################################
        #########################################################################################################
#         beta1 = nu*(cattle-infect_cow)**chi + (sheep-infect_sheep)**chi
#         beta1 = nu*(xi*(cattle-infect_cow)**chi + (sheep-infect_sheep)**chi)
#         beta = np.zeros(N)

#         for j in range(0,N):
#             beta[j] = beta1[j]*(np.sum((zeta*(infect_cow[I==2]**chi) +(infect_sheep[I==2]**chi))*kerneldist[I==2, j]))

#         prob_inf = (1 - np.exp(-beta)) 
#         unif = np.random.uniform(0, 1, N)

#         ####immunity####
#         for i in range(N):
#             if (I[i] == 4) or (I[i] == 3):
#                 cattle[i] = 0
#                 sheep[i] = 0
#                 prob_inf[i] = 0
#         for i in range(0,N):
#             if (unif[i] <= prob_inf[i] and I[i] == 0):
#                 cattle[i] = true_cattle[i]
#                 sheep[i] = true_sheep[i]
#                 I[i] =  1
#                 A[i, ] = [i, t, s[i], r[i], tr[i], q[i], 0, 0,0,0]
        #########################################################################################################
        ######################################## UPDATE STATES ##################################################
        #########################################################################################################

        ###################################### EXPOSED TO INFECTIOUS ############################################

        inf = A[:,0][A[:,1] + A[:,2] == t]

        I[inf.astype(np.int64)] = 2


        ####################### UPDATE NUMBER OF INFECTED ANIMALS WITHIN AN INFECTIOUS FARM #####################

        for i in inf: 
            i = int(i)
            if (random.random() < (sheep[i]/(sheep[i] +cattle[i])) and sheep[i]>=1 and infect_sheep[i]<sheep[i]):
                infect_sheep[i] += 1
            elif (cattle[i]>=1 and infect_cow[i]<cattle[i]):
                infect_cow[i] +=1

        ########################################### RING VACCINATION ###########################################
#         eff = np.random.beta(3, 10)
#         eff = np.random.beta(10,3)
#             eff = efficac
# #             print(eff)
#             trig = A[:,0][A[:,1] + A[:,2] + A[:,4] == t] #trigger vaccines
#     #         print(trig)

#             for i in range(len(trig)):
#                 n = [k for k in range(len(I)) if dist[trig[i].astype(np.int64), k] <RingCull/100]
#                 for j in range(len(n)):
#                     m = n[j]
#                     if A[m,7] == 0:
#                         VAC = np.append(VAC, m)
#                         travelled.append(dist[trig[i].astype(np.int64), m])
#     #         VAC = [x for _,x in sorted(zip([x for x in travelled],VAC))]
#             VAC, indices = np.unique(np.array(VAC, dtype=np.int), return_inverse=True)
#             VAC = VAC[indices]
#             if len(VAC) > 0:
#                 VAC = VAC.astype(np.int64)
#             if len(VAC)>Capacity:
#                 cost[VAC[0:Capacity]] += farm_cost[VAC[0:Capacity]]

#                 for j in range(len(VAC[0:Capacity])):
#                     m = VAC[j]
#                     A[m,0] = m
#                     A[m,7] = t
#                     if (I[m] == 0):

#                         if np.random.uniform(0,1) <eff:
#                             A[m, 8] = np.random.negative_binomial(6*28, 6*28/(12*28))
#                             A[m,9] = A[m,8] + t
#                             I[m] = 3
#                             cattle[m] = 0
#                             sheep[m] = 0
#                     elif (I[m] == 3):
#                         if np.random.uniform(0,1) < eff:
#                             immunity = np.random.negative_binomial(6*28, 6*28/(12*28))

#                             if (A[m,9]) < (immunity+t):
#                                 A[m,8] = immunity
#                                 A[m, 9] = immunity +t
#                             else:
#                                 A[m,8] = A[m,9] - t
#                     elif (I[m] ==4):
#                         if np.random.uniform(0,1) < eff:
#                             immunity = np.random.negative_binomial(6*28, 6*28/(12*28))
#                             if (A[m,6] )< (immunity +t):
#                                 A[m,8] = immunity
#                                 A[m,9] = immunity + t
#                                 I[m] = 3
#                                 A[m,6] = 0
#                                 cattle[m] = 0
#                                 sheep[m] = 0
#                 VAC = np.delete(VAC, range(0, Capacity), None)
#             else: 
#                 cost[0:len(VAC)] += farm_cost[0:len(VAC)]
#     #             eff = 1
#     #             eff = np.random.beta(3,10)
#     #             eff = np.random.beta(10,3)
#                 for j in range(len(VAC)):
#                     m = VAC[j]
#                     A[m, 0] = m
#                     A[m, 7] = t
#                     if (I[m] == 0):
#                         if np.random.uniform(0,1) <eff:
#                             A[m, 8] = np.random.negative_binomial(6*28, 6*28/(12*28))
#                             A[m, 9] = A[m, 8] +t
#                             I[m] = 3
#                             cattle[m] = 0
#                             sheep[m] = 0
#                     elif (I[m] == 3):
#                         if np.random.uniform(0,1) < eff:
#                             immunity = np.random.negative_binomial(6*28, 6*28/(12*28))
#                             if (A[m,9]) < (immunity +t):
#                                 A[m,8] = immunity
#                                 A[m,9] = immunity +t
#                             else:
#                                 A[m,8] = A[m,9] - t
#                     elif (I[m] ==4):
#                         if np.random.uniform(0,1) < eff:
#                             immunity = np.random.negative_binomial(6*28, 6*28/(12*28))
#                             if (A[m,6] )< (immunity +t):
#                                 A[m,8] = immunity
#                                 A[m,9] = immunity + t
#                                 I[m] = 3
#                                 A[m,6] = 0  
#                                 cattle[m] = 0
#                                 sheep[m] = 0
#                 VAC = np.delete(VAC, range(0, len(VAC)), None)


########################################### 25% RING VACCINATION ###########################################
        
#         eff = np.random.beta(10,3)
#         eff = efficac
#             print(eff)
        trig = A[:,0][A[:,1] + A[:,2] + A[:,4] == t] #trigger vaccines
#         print(trig)

        for i in range(len(trig)):

            n2 = [k for k in range(len(I)) if dist[trig[i].astype(np.int64), k] <1.0*RingCull/100]
            index=list(np.where(np.random.uniform(0,1,len(n2))<0.25*np.ones(len(n2)))[0])
            n=list(np.asarray(n2)[index])


#                 n = [k for k in range(len(I)) if dist[trig[i].astype(np.int64), k] <RingCull/100]
            for j in range(len(n)):
                m = n[j]
                if A[m,7] == 0:
                    VAC = np.append(VAC, m)
                    travelled.append(dist[trig[i].astype(np.int64), m])
#         VAC = [x for _,x in sorted(zip([x for x in travelled],VAC))]
        VAC, indices = np.unique(np.array(VAC, dtype=np.int), return_inverse=True)
        VAC = VAC[indices]
        if len(VAC) > 0:
            VAC = VAC.astype(np.int64)
        if len(VAC)>Capacity:
            eff = np.random.beta(2, 8)
            cost[VAC[0:Capacity]] += farm_cost[VAC[0:Capacity]]

            for j in range(len(VAC[0:Capacity])):
                m = VAC[j]
                A[m,0] = m
                A[m,7] = t
                if (I[m] == 0):

                    if np.random.uniform(0,1) <eff:
                        A[m, 8] = np.random.negative_binomial(6*28, 6*28/(12*28))
                        A[m,9] = A[m,8] + t
                        I[m] = 3
                        cattle[m] = 0
                        sheep[m] = 0
                elif (I[m] == 3):
                    if np.random.uniform(0,1) < eff:
                        immunity = np.random.negative_binomial(6*28, 6*28/(12*28))

                        if (A[m,9]) < (immunity+t):
                            A[m,8] = immunity
                            A[m, 9] = immunity +t
                        else:
                            A[m,8] = A[m,9] - t
                elif (I[m] ==4):
                    if np.random.uniform(0,1) < eff:
                        immunity = np.random.negative_binomial(6*28, 6*28/(12*28))
                        if (A[m,6] )< (immunity +t):
                            A[m,8] = immunity
                            A[m,9] = immunity + t
                            I[m] = 3
                            A[m,6] = 0
                            cattle[m] = 0
                            sheep[m] = 0
            VAC = np.delete(VAC, range(0, Capacity), None)
        else: 
            eff = np.random.beta(2, 8)
            cost[0:len(VAC)] += farm_cost[0:len(VAC)]
#             eff = 1
#             eff = np.random.beta(3,10)
#             eff = np.random.beta(10,3)
            for j in range(len(VAC)):
                m = VAC[j]
                A[m, 0] = m
                A[m, 7] = t
                if (I[m] == 0):
                    if np.random.uniform(0,1) <eff:
                        A[m, 8] = np.random.negative_binomial(6*28, 6*28/(12*28))
                        A[m, 9] = A[m, 8] +t
                        I[m] = 3
                        cattle[m] = 0
                        sheep[m] = 0
                elif (I[m] == 3):
                    if np.random.uniform(0,1) < eff:
                        immunity = np.random.negative_binomial(6*28, 6*28/(12*28))
                        if (A[m,9]) < (immunity +t):
                            A[m,8] = immunity
                            A[m,9] = immunity +t
                        else:
                            A[m,8] = A[m,9] - t
                elif (I[m] ==4):
                    if np.random.uniform(0,1) < eff:
                        immunity = np.random.negative_binomial(6*28, 6*28/(12*28))
                        if (A[m,6] )< (immunity +t):
                            A[m,8] = immunity
                            A[m,9] = immunity + t
                            I[m] = 3
                            A[m,6] = 0  
                            cattle[m] = 0
                            sheep[m] = 0
            VAC = np.delete(VAC, range(0, len(VAC)), None)


        ########################################### vaccine wears off ###########################################

        new_dose = A[:,0][A[:,7]+6*28 == t] #records when the farmer will get a new vaccine


        for j in new_dose:
            j = j.astype(np.int64)
            if (A[j,8] > 6*28) and (I[j]==3):
                A[j, 7] = 0
            else: 
                A[j, 7] = 0
                A[j, 8] = 0
                A[j,9] = 0
                if I[j] == 3:
                    I[j] = 0
                    cattle[j] = true_cattle[j]
                    sheep[j] = true_sheep[j]

        vaccinate_ends = A[:,0][A[:,9] == t]

        for j in vaccinate_ends:
            j = j.astype(np.int64)
            if I[j] ==3:
                I[j] =0
                cattle[j] = true_cattle[j]
                sheep[j] = true_sheep[j]

        ######################################### NO IMMUNITY ###################################################

#         rem = A[:,0][A[:,1] + A[:,2] + A[:,3] == t] #Move to S state once infectious period is over
#         infect_sheep[rem.astype(np.int64)] = 0
#         infect_cow[rem.astype(np.int64)] = 0
#         A[rem.astype(np.int64), ] = [0,0,0,0,0,0,0]
#         I[rem.astype(np.int64)] = 0

#         infected = [i for i in range(N) if (infect_cow+infect_sheep)[i]>0]
#         for farm in infected:
#             mu = 0.05
#             recover = np.random.uniform(0,1)
#             for animals in range(int(infect_cow[farm]+infect_sheep[farm])):
#                 if recover < mu and (infect_sheep[farm]+infect_cow[farm]) >=1 :
#                     if (random.random() < 0.5 and infect_sheep[farm]>=1 ):
#                         infect_sheep[farm] -= 1
#                     elif infect_cow[farm]>=1: 
#                         infect_cow[farm] -=1
#                     if infect_sheep[farm]+infect_cow[farm] ==0: #farm has recovered before its end period
#                         I[farm] = 0


        ######################################### IMMUNITY #######################################################

        rem = A[:,0][A[:,1] + A[:,2] + A[:,3] == t] #Move to IM state once infectious period is over
        for i in rem:
            i = i.astype(np.int64)
            if I[i] == 2:
                I[i] =4
                cattle[i] = 0
                sheep[i] = 0
                A[i, 6] = A[i, 5] +t
                infect_sheep[i] = 0
                infect_cow[i] = 0

        immune = A[:,0][A[:,1] + A[:,2] + A[:,3] + A[:,5] == t] #Move to S state once immune period is over
        I[immune.astype(np.int64)] = 0
        A[immune.astype(np.int64), ] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for k in immune:
            k = k.astype(np.int64)
            cattle[k] = true_cattle[k]
            sheep[k] = true_sheep[k]


        #### WITHIN FARM RECOVERY ####
        infected = [i for i in range(N) if (infect_cow+infect_sheep)[i]>0]
        for farm in infected:
            mu = 0.01
            recover = np.random.uniform(0,1)
            for animals in range(int(infect_cow[farm]+infect_sheep[farm])):
                if recover < mu and (infect_sheep[farm]+infect_cow[farm]) >=1 :
                    if (random.random() <(infect_sheep[farm]/(infect_sheep[farm]+infect_cow[farm])) and infect_sheep[farm]>=1 ):
                        infect_sheep[farm] -= 1
                    elif infect_cow[farm]>=1: 
                        infect_cow[farm] -=1
                    if infect_sheep[farm]+infect_cow[farm] ==0: 
                        I[farm] = 4
                        infect_sheep[farm] = 0
                        infect_cow[farm] = 0
                        cattle[farm] = 0
                        sheep[farm] = 0
                        A[farm, 6] = A[farm, 5] +t

        ############################################### PLOTS #####################################################

        infectind = [i for i in range(np.size(I)) if I[i]==2]
        print(infectind)
        susceptind = [i for i in range(np.size(I)) if I[i]==0]
        exposedind = [i for i in range(np.size(I)) if I[i] == 1]
        immuneind = [i for i in range(np.size(I)) if I[i] ==4]
        vaccineind = [i for i in range(np.size(I)) if I[i]==3]
        had_dose = [i for i in range(np.size(I)) if A[i,6]>0]
        if t>0:
            infNum += len(inf)

        numinf2 = len(inf)
        Dose.append(len(had_dose))
        #print('infect', sum(I==2), 'immue', sum(I==4), 'time', t, sum(I==0)+sum(I==1)+sum(I==2)+sum(I==3)+sum(I==4))
        Inf.append(len(infectind))
        Sus.append(len(susceptind))
        Exp.append(len(exposedind))
        Imm.append(len(immuneind))
        Vac.append(len(vaccineind))
        time_plot.append(t)



        xinfectplot = xcoord[infectind]
        yinfectplot = ycoord[infectind]
        xsusceptplot = xcoord[susceptind]
        ysusceptplot = ycoord[susceptind]
        xexposeplot = xcoord[exposedind]
        yexposeplot = ycoord[exposedind]
        ximmuneplot = xcoord[immuneind]
        yimmuneplot = ycoord[immuneind]
        xvaccineplot = xcoord[vaccineind]
        yvaccineplot = ycoord[vaccineind]
        ax.clear()

        ax1.clear()


#         ax1.plot(time_plot,Sus,label='Susceptible', c='yellowgreen')
        ax1.plot(time_plot,Inf,label='Infectious', c='r')
#         ax1.plot(time_plot, Imm, label = 'immune', c='b')
        ax1.plot(time_plot, Vac, label = 'vaccine', c='g')
        plt.xlabel('time')
        plt.ylabel('Number of Farms')
#         ax1.legend()


        ax.scatter(xsusceptplot, ysusceptplot, c='yellowgreen', marker='o', s=6, label='Susceptible')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(xinfectplot, yinfectplot, c='r', marker='o', s=10, label='Infectious')
        ax.scatter(ximmuneplot, yimmuneplot, c='b', marker='o', s=6, label='Immune')
        ax.scatter(xvaccineplot, yvaccineplot, c='g', marker = 'o', s = 6, label = 'vaccine')
        ax.axis([0, np.max(xcoord), 0, np.max(ycoord)])


        plt.title('Day {}, Infected: {}, Vaccine functional: {}'.format(t, np.size(xinfectplot), np.size(xvaccineplot)),fontsize = 12)
        ax.axis('scaled')
        ax.legend(loc = 'upper left', fontsize = 10.8,  markerscale= 3)

        cumInf[t] = cumInf[t-1] + infNum
#         print(cumInf)
        numInf[t] = numinf2

        if sum(I == 1) + sum(I == 2) == 0:
            a = cumInf[t-1]
            cumInf[t:] = a
            numInf[t:] = 0
            cumInfArray = np.vstack((cumInfArray, cumInf))
            InfArray = np.vstack((InfArray, numInf))
            endemicEFF[0] = 0
            cumInfEFF[0] = cumInf[599]
            maxEFF[0] = np.max(Inf)
            timeEFF.append(time_plot)
            infEFF.append(Inf)
            costEFF.append(cost)
            break
        if t == 599:
            cumInfEFF[0] = cumInf[599]
            endemicEFF[0] = np.mean(Inf[500:])
            maxEFF[0] = np.max(Inf)
            timeEFF.append(time_plot)
            infEFF.append(Inf)
            costEFF.append(cost)
            break
#         print('immune', count_imm, 'infect', count_inf)

        fig.canvas.draw()
    print(time.time()-start_time)
    Cumulative2.append(cumInfEFF)
    Endemic2.append(endemicEFF)
    Maximum2.append(maxEFF)
    Totalcost2.append(costEFF)
    Duration2.append(timeEFF)
    Infection2.append(infEFF)
    np.save('cumulativeinf', Cumulative2)
    np.save('Infect', Infection2)
    np.save('endemicinf', Endemic2)
    np.save('Maxinf', Maximum2)
    np.save('durationstime', Duration2)
    np.save('costsinf', Totalcost2)

# np.save('cumulative2', Cumulative2)
# np.save('Infection2', Infection2)
# np.save('endemic2', Endemic2)
# np.save('Max2', Maximum2)
# np.save('durations2', Duration2)
# np.save('costs2', Totalcost2)



