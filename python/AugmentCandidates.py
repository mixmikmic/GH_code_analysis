## Create new candidates file

import pandas as pd
import numpy as np

DATA_DIR = "/Volumes/data/tonyr/dicom/LUNA16/"
cand_path = 'CSVFILES/candidates_V2.csv'
annotations_path = 'CSVFILES/annotations.csv'

dfAnnotations = pd.read_csv(DATA_DIR+annotations_path).reset_index()
dfAnnotations = dfAnnotations.rename(columns={'index': 'candidate'})
dfCandidates = pd.read_csv(DATA_DIR+cand_path).reset_index()
dfCandidates = dfCandidates.rename(columns={'index': 'candidate'})

dfCandidates['diameter_mm'] = np.nan  # Set a new column and fill with NaN until we know the true diameter of the candidate

dfClass1 = dfCandidates[dfCandidates['class'] == 1].copy(deep=True)  # Get only the class 1 (they are the only ones that are labeled)

dfCandidates.shape

seriesuid = dfClass1['seriesuid'].unique()  # Get the unique series names (subjects)

for seriesNum in seriesuid:
    
    # Get the annotations for this candidate
    candAnnotations = dfAnnotations[dfAnnotations['seriesuid']==seriesNum]['candidate'].values
    candCandidates = dfClass1[dfClass1['seriesuid'] == seriesNum]['candidate'].values

    # Now loop through annotations to find closest candidate
    diameterArray = []

    for ia in candAnnotations: # Loop through the annotation indices for this seriesuid

        annotatePoint = dfAnnotations[dfAnnotations['candidate']==ia][['coordX', 'coordY', 'coordZ']].values

        closestDist = 10000

        for ic in candCandidates: # Loop through the candidate indices for this seriesuid

            candidatePoint = dfCandidates[dfCandidates['candidate']==ic][['coordX', 'coordY', 'coordZ']].values

            dist = np.linalg.norm(annotatePoint - candidatePoint)  # Find euclidean distance between points

            if dist < closestDist:  # If this distance is closer then update array
                closest = [ia, ic, 
                           dfAnnotations[dfAnnotations['candidate']==ia]['diameter_mm'].values[0],
                           dfAnnotations[dfAnnotations['candidate']==ia]['coordX'].values[0],
                           dfAnnotations[dfAnnotations['candidate']==ia]['coordY'].values[0],
                           dfAnnotations[dfAnnotations['candidate']==ia]['coordZ'].values[0]]
                closestDist = dist  # Update with new closest distance      

        diameterArray.append(closest)  
       
    # Update dfClass1 to include the annotated size of the nodule (diameter_mm)
    for row in diameterArray:
        dfClass1.set_value(row[1], 'diameter_mm', row[2])  
        dfClass1.set_value(row[1], 'coordX_annotated', row[3])
        dfClass1.set_value(row[1], 'coordY_annotated', row[4])
        dfClass1.set_value(row[1], 'coordZ_annotated', row[5])
        

dfClass1.iloc[:10,:]

del dfCandidates['diameter_mm']

dfOut = dfCandidates.join(dfClass1[['candidate', 'diameter_mm', 'coordX_annotated', 'coordY_annotated', 'coordZ_annotated']], on='candidate', rsuffix='_r')
del dfOut['candidate_r']
del dfOut['candidate']

dfOut.to_csv('candidates_with_annotations.csv', index=False)



