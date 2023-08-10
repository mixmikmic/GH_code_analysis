import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
get_ipython().magic('matplotlib inline')

# This funtion produces a dictionary that maps each GO term to a tissue in GTEX (if we have connected the two via BTO)
def map_GO_to_GTEX():
    inputFilename = '../data/GO_terms_final_gene_counts.txt'
    GO_list_file = open(inputFilename)
    GO_list = np.loadtxt(GO_list_file,skiprows=2,usecols=[0],dtype='S10',delimiter='\t')
    
    inputFilename = '../data/Tissue_Name_Mappings.csv'
    tissue_data = pd.read_csv(inputFilename,header=None)
    map_BTO_to_GTEX = defaultdict(list)

    for index,row in tissue_data.iterrows():
        GTEX_tissue = row[0]
        BTO_tissues = row[1:]
        for tissue in BTO_tissues.dropna():
            map_BTO_to_GTEX[tissue].append(GTEX_tissue)



    inputFilename = '../data/BTO_GO.csv'
    BTO_data = pd.read_csv(inputFilename,skiprows=[0])
    map_GO_to_GTEX = defaultdict(list)

    for index,row in BTO_data.iterrows():
        tissue = row[1]
        if tissue in map_BTO_to_GTEX:
            GO_IDs = row[2:]
            for GO_ID in GO_IDs.dropna():
                if GO_ID in GO_list:
                    map_GO_to_GTEX[GO_ID] = map_GO_to_GTEX[GO_ID] + map_BTO_to_GTEX[tissue]

    #inputFile.close()
    return map_GO_to_GTEX

# function takes result file and returns GO_ID and array of coefficients
def get_coeff(input_file):
    results = open(input_file)
    nextline = 0
    for line in results:
        data = line.split()
        if data[1] == 'Prediction':
            GO_ID = data[len(data)-1]
        elif data[1] == 'Coefficients:':
            nextline = 1
        elif nextline == 1:
            coeffs = (data)
            break
    return [GO_ID, coeffs]

# function takes GO term, array of coefficients, and (optional) highlighted tissues, and plots accordingly
def plot_coeff(GO_ID, coeffs, highlight = []):
    abs_coeff = [abs(float(c)) for c in coeffs]
    samples = open('../data/samples_to_tissues_map.txt')
    tissue_type = np.loadtxt(samples,dtype='S40',delimiter='\t')
    tissue_list = pd.unique(tissue_type[:,2]) # pandas maintains correct order, numpy doesn't
    
    plt.figure(figsize=(18, 6))
    plt.margins(0.01)
    ax = plt.gca()
    ax.xaxis.grid(which='both')
    plt.xticks(range(len(tissue_list)), tissue_list, rotation='vertical')
    
    x = np.array([np.where(tissue_list == tissue)[0][0] for tissue in tissue_type[:,2]])
    y = np.array(abs_coeff)
    
    if highlight:
        ix = np.in1d(tissue_type[:,2], highlight)
        ind = np.where(ix)[0]
        x_part = x[ind]
        y_part = y[ind]
        plt.plot(x_part, y_part, 'ro')
        x_part = np.delete(x,ind)
        y_part = np.delete(y,ind)
        plt.plot(x_part,y_part,'bo')
    else:
        plt.plot(x, coeffs, 'bo')
    
    plt.show()

# Create dictionary that maps GTEX tissues to GO terms and create histogram

GOmap = map_GO_to_GTEX()
GTEXmap = defaultdict(list)
for GO_ID,tissues in GOmap.items():
    for tissue in tissues:
        if GO_ID not in GTEXmap[tissue]:
            GTEXmap[tissue].append(GO_ID)
num_GO = [len(array) for array in GTEXmap.values()]
index_order = np.argsort(num_GO) # sort in order
#index_order = range(len(num_GO)) # leave in order of dictionary
tissues = np.array(GTEXmap.keys())[index_order]
values = np.array(num_GO)[index_order]

plt.figure(figsize=(18, 6))
plt.margins(0.01)
plt.bar(range(len(tissues)), values, align='center')
plt.xticks(range(len(tissues)), tissues, rotation='vertical')
plt.show()

[GO_ID, coeffs] = get_coeff('../GO_prediction/Results/full_results_all_tissues_loss_l1_neg_0/logreg_GO:0071827.txt')
GOmap = map_GO_to_GTEX()
tissues = GOmap[GO_ID]
print tissues
plot_coeff(GO_ID,coeffs,highlight = tissues)

[GO_ID, coeffs] = get_coeff('../GO_prediction/Results/full_results_all_tissues_loss_l1_neg_0/logreg_GO:0070527.txt')
GOmap = map_GO_to_GTEX()
tissues = GOmap[GO_ID]
print tissues
#tissues = [['Whole Blood','Lung']]
plot_coeff(GO_ID,coeffs,highlight = tissues)

[GO_ID, coeffs] = get_coeff('../GO_prediction/Results/full_results_all_tissues_loss_l1_neg_0/logreg_GO:0003009.txt')
GOmap = map_GO_to_GTEX()
tissues = GOmap[GO_ID]
print tissues
plot_coeff(GO_ID,coeffs,highlight = tissues)



