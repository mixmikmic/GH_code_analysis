get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
from biopandas.pdb import PandasPdb as PandasPdb
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

key_atoms = ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C4A', 'C4X', 'N5', 'C5A', 'C5X', 'C6', 'C7', 'C7M', 'C8', 'C8M', 'C9', 'C9A', 'N10', 'C10']
key_res = ['FMN', 'FAD']
ambiguous_atom = ['C4A', 'C4X', 'C5A', 'C5X']

#Choose flavoprotein and angstrom limit
flavoprotein = input("Please enter flavoprotein (Example: 1pbe): ")
bioassembly = input("Please enter biological assembly number. Enter 0 if not reading biological assembly: ")
angstrom_limit = float(input("Please enter desired ångström limit: "))

#If 0 is entered, then read normal PDB file. If a number is given, then read the given biological assembly
if bioassembly == '0':
    ppdb = PandasPdb().fetch_pdb(flavoprotein + ".pdb")
else:
    ppdb = PandasPdb().fetch_pdb(flavoprotein + ".pdb" + bioassembly)

#Create dataframes from chosen pdb file
df_atom = ppdb.df['ATOM']
df_hetatm = ppdb.df['HETATM']
df = pd.concat([df_atom, df_hetatm])
df = df.reset_index(drop=True)

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
        
    #This is to reindex the indices of distance_het_dict from df_hetatm indexes to the combined df indexes
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

def label(distance_atom_dict, distance_het_dict):
    new_atom = {}
    new_het = {}
    
    for k,v in distance_atom_dict.items():
        temp = []
        for i in v.index:
            temp.append(((df.residue_name[i] + str(df.residue_number[i]) + ":" + df.atom_name[i]), v[i]))
        new_atom[df.residue_name[k] + str(df.residue_number[k]) + ":" + df.atom_name[k]] = temp
    
    for k,v in distance_het_dict.items():
        temp2= []
        for i in v:
            temp2.append(((df.residue_name[i[0]] + str(df.residue_number[i[0]]) + ":" + df.atom_name[i[0]]), i[1]))
        new_het[df.residue_name[k] + str(df.residue_number[k]) + ":" + df.atom_name[k]] = temp2
    
    return((new_atom, new_het))

#Outputs a CSV file of the distances
def make_csv(atom, het):
    df_distance = pd.DataFrame(columns=['Reference_Atom', 'Target_Atom', 'Distance'])

    a_list = []
    b_list = []
    c_list = []
    for k,v in atom.items():
        for target in v:
            a_list.append(k)
            b_list.append(target[0])
            c_list.append(target[1])

    for k,v in het.items():
        for target in v:
            a_list.append(k)
            b_list.append(target[0])
            c_list.append(target[1])

    df_distance['Reference_Atom'] = a_list
    df_distance['Target_Atom'] = b_list
    df_distance['Distance'] = c_list
    df_distance = df_distance.set_index('Reference_Atom')
    df_distance = df_distance.sort_index()

    if bioassembly != "0":
        df_distance.to_csv(flavoprotein + ".pdb" + bioassembly + "_distances.csv", sep=',')
    else:
        df_distance.to_csv(flavoprotein + ".pdb" + "_distances.csv", sep=',')

def main():
    #Create empty dictionaries for distances
    distance_atom_dict = {}
    distance_het_dict = {}
    
    #Call 'find_distance' function and set empty dictionaries equal to output of the function
    distances = find_distance(distance_atom_dict, distance_het_dict, angstrom_limit)
    distance_atom_dict = distances[0]
    distance_het_dict = distances[1]
    
    #Call 'label' function and update dictionaries to labeled dictionaries
    relabeled = label(distance_atom_dict, distance_het_dict)
    distance_atom_dict = relabeled[0]
    distance_het_dict = relabeled[1]
    
    #Create CSV file
    make_csv(distance_atom_dict, distance_het_dict)

if __name__ == "__main__":
    main()



