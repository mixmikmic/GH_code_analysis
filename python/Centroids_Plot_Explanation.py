## Import pipeline
import analysis3 as a3

## For s275 to ara3:
points_path = 's275_to_ara3.csv'
output_path = 'output/s275_centroids_by_size.html'
p, uniq = a3.get_regions(points_path, "atlas/ara3_annotation.nii", "points/" + 's275_to_ara3_run' + "_regions.csv");

points_path = 'points/s275_to_ara3_run_regions.csv'
fig = a3.generate_scaled_centroids_graph('s275', points_path, uniq, output_path = output_path)

plotly.offline.init_notebook_mode() # run at the start of every ipython notebook
iplot(fig, filename = "outputtest.html")

## Comparison for the original region_graph:
points_path = 'points/s275_to_ara3_run_regions.csv'
fig2, output = a3.generate_region_graph("s275_to_ara3", points_path);

iplot(fig2, filename = "outputtest2.html")

import numpy as np
import os

import plotly
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly import tools
import csv
import seaborn as sns

import pickle

import requests
import json

def generate_scaled_centroids_graph(token, points_path, unique_list, output_path=None):
    """
    Generates the plotly centroid html file with proper scaling of the centroid determined by the number of bright spots from the csv file.
    :param points_path: Filepath to points csv.
    :param output_path: Filepath for where to save output html.
    :param resolution: Resolution for spacing (see openConnecto.me spacing)
    :return:
    """
    
    # Type in the path to your csv file here
    thedata = None
    thedata = np.genfromtxt(points_path,
        delimiter=',', dtype='int', usecols = (0,1,2,4), names=['a','b','c', 'region'])
    
    # Save the names of the regions
    """
    Load the CSV of the ARA with CCF v3: in order to generate this we use the ARA API.
    We can download a csv using the following URL:
    http://api.brain-map.org/api/v2/data/query.csv?criteria=model::Structure,rma::criteria,[ontology_id$eq1],rma::options[order$eq%27structures.graph_order%27][num_rows$eqall]
    
    Note the change of ontology_id$eq27 to ontology_id$eq1 to get the brain atlas.
    """
    ccf_txt = 'natureCCFOhedited.csv'

    ccf = {}
    with open(ccf_txt, 'rU') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # row[0] is ccf atlas index, row[4] is string of full name
            ccf[row[0]] = row[4];
    
    # Sort the csv file by the last column (by order = 'region') using ndarray.sort
    sort = np.sort(thedata, order = 'region');
    
    # Save the unique counts
    unique = [];

    for l in sort:
        unique.append(l[3])

    uniqueNP = np.asarray(unique)
    allUnique = np.unique(uniqueNP)
    numRegionsA = len(allUnique)
    
    """
    First we download annotation ontology from Allen Brain Atlas API.
    It returns a JSON tree in which larger parent structures are divided into smaller children regions.
    For example the "corpus callosum" parent is has children "corpus callosum, anterior forceps", "genu of corpus callosum", "corpus callosum, body", etc
    """

    url = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
    jsonRaw = requests.get(url).content
    jsonDict = json.loads(jsonRaw)

    """
    Next we collect the names and ids of all of the regions.
    Since our json data is a tree we can walk through it in arecursive manner.
    Thus starting from the root...
    """
    root = jsonDict['msg'][0]
    """
    ...we define a recursive function ...
    """

    leafList = []

    def getChildrenNames(parent, childrenNames={}):
        if len(parent['children']) == 0:
            leafList.append(parent['id'])

        for childIndex in range(len(parent['children'])):
            child = parent['children'][childIndex]
            childrenNames[child['id']] = child['name']

            childrenNames = getChildrenNames(child, childrenNames)
        return childrenNames

    """
    ... and collect all of the region names in a dictionary with the "id" field as keys.
    """

    regionDict = getChildrenNames(root)

    # Store most specific data
    specificRegions = [];

    for l in sort:
        if l[3] in leafList:
            specificRegions.append(l)

    # Find all unique regions of brightest points (new)
    uniqueFromSpecific = [];
    final_specific_list = [];

    for l in specificRegions:
        uniqueFromSpecific.append(l[3])
    
    for l in sort:
        if l[3] in uniqueFromSpecific:
            final_specific_list.append(l)

    # Convert to numpy and save specific lengths
    uniqueSpecificNP = np.asarray(uniqueFromSpecific)
    allUniqueSpecific = np.unique(uniqueSpecificNP)
    numRegionsASpecific = len(allUniqueSpecific)
    specificRegionsNP = np.asarray(specificRegions)

    print "Total number of unique ID's:"
    print numRegionsASpecific  ## number of specific regions

    # Save a copy of this sorted/specific csv in the points folder
    if not os.path.exists('points'):
        os.makedirs('points')
        
    np.savetxt('points/' + str(token) + '_regions_sorted.csv', sort, fmt='%d', delimiter=',')
    np.savetxt('points/' + str(token) + '_regions_sorted_specific.csv', specificRegionsNP, fmt='%d', delimiter=',')
    
    # Find the centroids of each region
    sorted_regions = np.sort(allUniqueSpecific);

    current_region = sorted_regions[0];
    
    i = 0;
    x = [];
    y = [];
    z = [];
    centroids = {};
    regions_dict = {};

    for row in final_specific_list:
        if row[3] == current_region:
            # Append x, y, z to appropiate list
            x.append(row[0]);
            y.append(row[1]);
            z.append(row[2]);
        else:
            # Store in centroids dictionary with key current_region the average x, y, z position.
            # Also store the number of points.
            centroids[current_region] = [np.average(x), np.average(y), np.average(z), len(x)];

            # Increment i, change current_region
            i = i + 1;
            current_region = sorted_regions[i]

            # Set x, y, z to new row values;
            x = [row[0]];
            y = [row[1]];
            z = [row[2]];

    # Store last region averages also!
    centroids[current_region] = [np.average(x), np.average(y), np.average(z), len(x)];
    
    print "length keys:"
    print len(centroids.keys());
    
    # Save a copy of the dictionary as a pickle.
    with open(token + '_centroids_dict.pickle', 'wb') as handle:
        pickle.dump(centroids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    trace = [];
    for l in specificRegionsNP:
        if str(l[3]) in ccf.keys():
            trace = ccf[str(l[3])]
    
    # Set color pallete to number of specific regions
    current_palette = sns.color_palette("husl", numRegionsA)
    
    i = 0;

    data = [];
    
    for key in centroids.keys():
        if str(key) not in ccf.keys():
            print "Not found in ara3 leaf nodes: " + str(key);
            i = i + 1;
            
        else:
            current_values_list = centroids[key];

            tmp_col = current_palette[i];
            tmp_col_lit = 'rgb' + str(tmp_col);

            trace_scatter = Scatter3d(
                x = [current_values_list[0]],
                y = [current_values_list[1]],
                z = [current_values_list[2]],
                mode = 'markers',
                name = ccf[str(key)],
                marker = dict(
                    size= np.divide(float(current_values_list[3]), 10000) * 500,
                    color = tmp_col_lit,     # set color to an array/list of desired values
                    colorscale = 'Viridis',  # choose a colorscale
                    opacity = 0.5
                )
            )

            data.append(trace_scatter)
            i = i + 1;

    layout = Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        paper_bgcolor='rgb(0,0,0)',
        plot_bgcolor='rgb(0,0,0)'
    )

    fig = Figure(data=data, layout=layout)

    if not os.path.exists('output'):
        os.makedirs('output')

    if output_path != None:
        plotly.offline.plot(fig, filename=output_path)

