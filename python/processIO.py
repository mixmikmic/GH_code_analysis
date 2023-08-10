import pandas as pd
import scipy.io as sio #to read matlab tables
from glob import glob
import numpy as np

gtap_sectors = pd.read_csv("GTAP_sectors.csv", usecols=[2], squeeze=True)
"  --  ".join(gtap_sectors.tolist())

ener_sector =  ['Coal', 'Oil', 'Gas','Petroleum, coal products', "Electricity", 'Gas manufacture, distribution' ]

#Binning in 4 sectors

sec_num_to_agg = pd.read_excel("GTAP_sector_groups.xlsx")
sec_num_to_agg["gtap_sector"]=sec_num_to_agg.Number.replace(gtap_sectors)
gtap_sectors_to_4sectors= sec_num_to_agg[["gtap_sector","4sectors"]].drop_duplicates().set_index("gtap_sector").squeeze()
gtap_sectors_to_4sectors.sample(6)

#Binning in manuf/others

gtap_sectors_to_2sectors = gtap_sectors_to_4sectors.replace(dict(agriculture="other", extract="other", service="other" ))
gtap_sectors_to_2sectors.name = "2sectors"
gtap_sectors_to_2sectors.sample(6)

def process_matfile(matfilepath, output_measure="Y"):
    #Unpackes the dictionaries provided by K. What should be used as total output is ambiguus, because the IO tables do not match line by line
        
    matfile = sio.loadmat(matfilepath)
              
    C = pd.Series(matfile["y_HH"].flatten(), index=gtap_sectors)  #consumption
    
    Y = pd.Series(matfile["x"].flatten(), index=gtap_sectors) #sectoral output
    
    if "Value_added" in matfile.keys():
        VA= pd.DataFrame(matfile["Value_added"], columns=gtap_sectors) #value_added
    else:
        VA = None 
    
    Inv = pd.Series(matfile["y_Cap"].flatten(), index=gtap_sectors) #caiptal formation
    
    G = pd.Series(matfile["y_Gov"].flatten(), index=gtap_sectors) #government spending
    
    X = pd.Series(matfile["Exp"].flatten(), index=gtap_sectors) #exports
    
    if "Imp" in matfile.keys():
        M = pd.Series(matfile["Imp"].flatten(), index=gtap_sectors) #exports
    else:
        M = None

    AX = pd.DataFrame(matfile["Z_dom"], index=gtap_sectors, columns=gtap_sectors) #is the domestic inter-sectoral flows

    if M is None:
        M=pd.Series(0, index=gtap_sectors)
        print("M set to 0")
    if VA is None:
        VA=pd.Series(0, index=gtap_sectors)
        print("VA set to 0")
    
    tot_output = AX.sum(axis=0)+Inv+G+X+C
    tot_input = AX.sum(axis=1)+VA.sum()+M
    
    if output_measure == "Y":
        A = AX.div(Y,axis=1) # i think this is the matrice of IO coefficients     
        Q=Y
    elif output_measure=="tot_output":
        A = AX.div(tot_output)
        Q=tot_output
    elif output_measure=="tot_input":
        A = AX.div(tot_input)
        Q=tot_input
       
    
    return C,Y, Q,VA,Inv,G,X,M,AX,A 
    
    

for matfilename in glob("LAcountries/*.mat"):
    
    cur_economy_key = matfilename.split("\\")[-1].split(".mat")[0]

    print("\n"+cur_economy_key)

    for output_measure in ["tot_input", "tot_output"]:
        print("Q=",output_measure)
        C,Y, Q,VA,Inv,G,X,M,AX,A = process_matfile(matfilename, output_measure=output_measure)
        
        
        
        if A.isnull().sum().sum()>0:
            print("Nans in ", A[A.isnull().sum()>0].index.tolist(),"converted to 0")
            A = A.fillna(0)

        print("Q/Y TOT",(Q.sum()/Y.sum())) 
        print("Q/Y line by line",(Q/Y).abs().argmax(),(Q/Y).abs().max())  



                

## Energy efficiency 
df=pd.DataFrame()
for matfilename in glob("LAcountries/*.mat"):
#     print("=========\n"+matfilename)
    
    cur_economy_key = matfilename.split("\\")[-1].split(".mat")[0]
    
    for output_measure in ["tot_input", "tot_output", "Y"]:
#         print("\n",output_measure)
        C,Y, Q,VA,Inv,G,X,M,AX,A = process_matfile(matfilename, output_measure=output_measure)
        EIdetailed = A.ix[ener_sector].sum()
        EI =(Q*EIdetailed).groupby(gtap_sectors_to_2sectors).sum()/Q.groupby(gtap_sectors_to_2sectors).sum()
        shQ =(Q).groupby(gtap_sectors_to_2sectors).sum()/Q.sum()
        df= df.append(pd.DataFrame([cur_economy_key, EI.other, EI.manufacture, shQ.other, shQ.manufacture, output_measure], index = ["economy","EI_other", "EI_manuf", "sh_other", "sh_manuf", "scenario"]).T, ignore_index=True)

        
df.set_index(["economy","scenario"]).unstack("scenario")       

C,Y, Q,VA,Inv,G,X,M,AX,A = process_matfile("LAcountries/ECU.mat")
AX["Electricity"].sort_values(ascending=False)

AX.sort_values(by="Electricity", ascending=False)



X-X.drop("Coal")



