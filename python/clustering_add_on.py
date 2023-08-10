get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
from biopandas.pdb import PandasPdb as PandasPdb
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy as sp
import scipy.spatial.distance as spd
import scipy.cluster.hierarchy as sph
import sklearn as sk
import sklearn.metrics as skm
import sklearn.cluster as skc
import sklearn.decomposition as skd
import sklearn.mixture as skmix
import sklearn.manifold as manifold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns

key_atoms = ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C4A', 'C4X', 'N5', 'C5A', 'C5X', 'C6', 'C7', 'C7M', 'C8', 'C8M', 'C9', 'C9A', 'N10', 'C10']
key_res = ['FMN', 'FAD']
ambiguous_atom = ['C4A', 'C4X', 'C5A', 'C5X']

#Choose flavoprotein and angstrom limit. Choose if loading a biological assembly.
flavoprotein = input("Please enter flavin molecule(Example: 1pbe): ")
bioassembly = input("Please enter biological assembly number. Enter 0 if not reading biological assembly: ")
angstrom_limit = float(input("Please enter desired ångström limit: "))

#If 0 is entered, then read normal PDB file. If a number is given, then read the given biological assembly
if bioassembly == '0':
    #File path connects to Dr. Palfey's hard drive. Change file path to where PDB files are stored.
    ppdb = PandasPdb().read_pdb("/Users/davidkim/Documents/Senior/SI 485/pdb_file/%s.pdb" % flavoprotein)
else:
    ppdb = PandasPdb().read_pdb("/Users/davidkim/Documents/Senior/SI 485/pdb_file/%s.pdb" % flavoprotein + bioassembly)
    
#Create dataframes from chosen pdb file
df_atom = ppdb.df['ATOM']
df_hetatm = ppdb.df['HETATM']
df = pd.concat([df_atom, df_hetatm])
df = df.reset_index(drop=True)

#Dataframe of the codes for atom types
df_codes = pd.read_csv("chemical_codes.csv")

#Dataframe of the chemical names of the codes
df_chemicals = pd.read_csv("chemical_types.csv")

#Loads in DataFrame of existing data from previous clustering script
df_frequency = pd.read_csv("BA" + bioassembly + "_" + str(angstrom_limit) + "Å_clustering_flavoproteins.csv", index_col=0)

#Find index number of N5 to check which of the ambiguous atoms are isoalloxazines
def check_ambiguous_atoms():
    index_N5 = []
    for i in range(len(df_hetatm)): #Use df_hetatm instead of df because distance function reads only separated dataframe
        if df_hetatm.atom_name[i]== 'N5':
            index_N5.append(i)
        else:
            continue
    return(index_N5)

#Find index number of C1 to exclude distant between isoalloxazine and C1 atom
def check_c1_atom():
    index_C1 = []
    for i in range(len(df_hetatm)): #Use df_hetatm instead of df because distance function reads only separated dataframe
        if df_hetatm.atom_name[i] == "C1'":
            index_C1.append(i)
        else:
            continue
    return(index_C1)

def find_distance(distance_atom_dict, distance_het_dict, angstrom_limit):    
    i = 0
    for i in range(len(df)):
        if ((df.residue_name[i] in key_res) and (df.atom_name[i] in key_atoms)):
            reference_point = (df.x_coord[i], df.y_coord[i], df.z_coord[i])
            distances_atm = ppdb.distance(xyz=reference_point, records='ATOM')
            distances_het = ppdb.distance(xyz=reference_point, records='HETATM')
            for c1 in check_c1_atom(): #Checking to see if C1 atom is in isoalloxazine
                if (df.residue_number[i] == df_hetatm.residue_number[c1]):
                    distances_het = distances_het.drop(c1)
                else:
                    distances_het = distances_het
            if df.atom_name[i] in ambiguous_atom: #Checking to see which ambiguous atom is correct
                for ind in check_ambiguous_atoms():
                    if distances_het[ind] < 2.0: #If less than 2 Angstroms away, by recommendation of Bruce
                        distance_het_dict[i] = distances_het[distances_het <= angstrom_limit]
                        distance_atom_dict[i] = distances_atm[distances_atm <= angstrom_limit]
                        break
                    else:
                        continue
            else:
                distance_het_dict[i] = distances_het[distances_het <= angstrom_limit]
                distance_atom_dict[i] = distances_atm[distances_atm <= angstrom_limit]
                
    #Remove intra-isoalloxazine distances
    for k,v in distance_het_dict.items():
        for iso in v.index:
            if ((df_hetatm.atom_name[iso] in key_atoms) and (df_hetatm.residue_number[iso] == df.residue_number[k])):
                v = v.drop(iso)
        distance_het_dict[k] = v
        
    #This is to relabel the indices of distance_het_dict from df_hetatm indexes to the combined df indexes
    pairs = {}
    for k,v in distance_het_dict.items():
        lst = []
        if len(v.index) != 0:
            for targ in v.index:
                check_res_numb = df_hetatm.residue_number[targ]
                check_atom_name = df_hetatm.atom_name[targ]
                new_targ = df.index[(df.residue_number == check_res_numb) & (df.atom_name == check_atom_name)]
                lst.append((new_targ.values[0], v[targ]))
            pairs[k] = lst
    
    distance_het_dict = pairs
    
    return (distance_atom_dict, distance_het_dict)

def distance_dataframe(distance_atom_dict, distance_het_dict):
    ref_atom_list = []
    target_atom_list = []
    distance_list = []

    for k,v in distance_atom_dict.items():
        for target, dist in v.iteritems():
            ref_atom_list.append(k)
            target_atom_list.append(target)
            distance_list.append(dist)

    for k,v in distance_het_dict.items():
        for i in v:
            ref_atom_list.append(k)
            target_atom_list.append(i[0])
            distance_list.append(i[1])

    df_dist = pd.DataFrame()
    df_dist['isoalloxazine'] = ref_atom_list
    df_dist['target_atom'] = target_atom_list
    df_dist['distance'] = distance_list
    
    #This is to find the associated chemical code of the target atom/heteroatm
    code_list = []
    for i in df_dist['target_atom']:
        temp = df_codes.index[(df_codes.Residue == df.residue_name[i]) & (df_codes.Atom == df.atom_name[i])]
        if len(temp) != 0:
            code_list.append(df_codes.Code[temp[0]])
        else:
            code_list.append(0) #Every atom is not represented in the code doc Dr. Palfey gave us. 0 for now.

    df_dist['chemical_codes'] = code_list
    
    #This is to find the associated chemical type of the target atom/heteroatm
    chemical_list = []
    for i in df_dist['chemical_codes']:
        temp = df_chemicals.index[df_chemicals.Code == i]
        try:
            chemical_list.append(df_chemicals.Label[temp[0]])
        except:
            chemical_list.append("N/A")
    
    df_dist['chemical_type'] = chemical_list
    
    return(df_dist)

def kmeans_cluster(df):
    #Determine amount of clusters
    number_of_cluster = np.round(np.sqrt(len(df)/2))
    
    kmeans_model = skc.KMeans(n_clusters=int(number_of_cluster))
    kmeans_model.fit(df)
    centroids = kmeans_model.cluster_centers_
    
    #Create column for K-means label for each flavoprotein
    df['cluster_labels_kmeans'] = kmeans_model.labels_
    
    return(df)

#Create a dendrogram using heirarchical clustering
def create_dendrogram(df_value):
    dist_value = spd.squareform(spd.pdist(df_value,metric='euclidean'))
    dist_value.shape
    Z_lang = sph.linkage(dist_value,method='complete') #Choose between single, complete, ward for method
    _ = sph.dendrogram(Z_lang, labels=df_value.index, orientation='left')
    plt.xlabel('Data Points')
    plt.ylabel('Flavoproteins')
    plt.suptitle('Dendrogram: Chemical-type count at ' + str(angstrom_limit) + ' limit', 
                 fontweight='bold', fontsize=14);

def append_dataframe(collect_dict):
    #Fill dataframe with 'collect_dict' information
    temp_df = pd.DataFrame(collect_dict)
    temp_df = temp_df.transpose()
    temp_df = temp_df.astype(int)
    temp_df = temp_df.fillna(0)
    
    if flavoprotein not in df_frequency.index:
        df_combined = pd.concat([df_frequency, temp_df], axis=0)
        df_combined = df_combined.fillna(0)
        df_combined = df_combined.astype(int)
        df_combined = kmeans_cluster(df_combined)
    else:
        return(df_frequency)

    return(df_combined)

def main():    
    #Create empty dictionaries for distances
    distance_atom_dict = {}
    distance_het_dict = {}
    
    #Call 'find_distance' function and set empty dictionaries equal to output of the function
    distances = find_distance(distance_atom_dict, distance_het_dict, angstrom_limit)
    distance_atom_dict = distances[0]
    distance_het_dict = distances[1]
    
    #Create a datagrame that holds the distances and chemical codes from the distance_dataframe function
    df_distance = distance_dataframe(distance_atom_dict, distance_het_dict)
    
    collect_dict = {}
    collect_dict[flavoprotein] = df_distance.chemical_type.value_counts()
    
    df_kmeans = append_dataframe(collect_dict)
    df_kmeans = df_kmeans.sort_values(by=['cluster_labels_kmeans'])
    df_kmeans.to_csv("BA" + bioassembly + "_" + str(angstrom_limit) + "Å_clustering_flavoproteins.csv", sep=',')
    display(df_kmeans)
    
    display(create_dendrogram(df_kmeans))

if __name__ == "__main__":
    main()



