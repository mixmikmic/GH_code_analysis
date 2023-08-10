import os
import numpy as np
from datetime import datetime
from pandas import DataFrame, concat
from scipy.io import loadmat

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Main loop.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

df = []
for cohort in ['behav','fmri']:
    
    ## Locate files.
    raw_dir = os.path.join('raw', cohort)
    files = os.listdir(raw_dir)

    ## Iterate over files.
    for f in files:
        
        ## Load and extract data.
        mat = loadmat(os.path.join(raw_dir, f))
        data = mat['Data'].squeeze()
    
        ## Extract subject/datetime info.
        subject, date, time = f.replace('.mat','').split('_')
        date = '-'.join(['%0.2d' %int(s) for s in date.split('-')])
        time = '-'.join(['%0.2d' %int(s) for s in time.split('-')])
        dt = datetime.strptime('%s %s' %(date,time), '%Y-%m-%d %H-%M-%S')

        ## Iterate over blocks.
        for blockno, block in enumerate(data):

            ## Separate data categories
            ## ------------------------
            ## - outcome: binary array indicating win (1) or loss (0).
            ## - onsets: list of arrays with task event onsets.
            ## - choice: integer array indicating machine chosen.
            ## - ratings: list of arrays with mood/probability ratings.
            ## - info: list of arrays of block information.
            outcome, onsets, choice, ratings, info = block

            ## Define machine identities
            ## -------------------------
            ## Machine stimulus and outcome probability are counterbalanced
            ## across participants. In the mat files, machines are labelled
            ## according to their stimulus, not their outcome probability.
            ## This section of code inverts this (i.e. identifies machine
            ## by outcome probability). This ultimately results in the 
            ## categorization scheme:
            ## - 20% probability = machines [1, 4, 7]
            ## - 40% probability = machines [2, 5, 8]
            ## - 60% probability = machines [3, 6, 9]
            _, presentation, identities = [arr.squeeze() for arr in info.squeeze().tolist()]
            if blockno < 3: identities += blockno * identities.size
            else: identities += np.repeat(np.arange(0,9,3),3).astype(identities.dtype)
            M1, M2 = identities[presentation-1]  

            ## Compute reaction times
            ## ----------------------
            ## Reaction times are simply computed from the onsets data.
            ## The second and third rows of the onsets matrix are
            ## stimulus and response onsets, respectively. Subtracting
            ## the second from the first returns the reaction time.
            onsets = np.array(onsets.squeeze().tolist()[:5]).squeeze()
            RT = onsets[2] - onsets[1]

            ## Compute outcome
            ## ---------------
            ## Outcome is already a binary vector (win = 1, lose = 0).
            ## This is simply multiplied by 0.25 for Blocks 1-3, and 
            ## by 0.50 for Blocks 4. 
            if blockno < 3: outcome = outcome * 0.25
            else: outcome = outcome * 0.50

            ## Assemble DataFrame
            ## ------------------
            ## Stores the following information: subject ID, block number, 
            ## trial number, machine 1/2, choice, reaction time, outcome.
            d = dict(Cohort = np.repeat(cohort, RT.size),
                     Datetime = np.repeat(dt, RT.size),
                     Block = np.repeat(blockno+1, RT.size),
                     Trial = np.arange(RT.size)+1,
                     M1 = M1, M2 = M2, Choice = choice.squeeze(),
                     RT = RT, Outcome = outcome.squeeze())
            df.append(DataFrame(d, columns=('Cohort','Datetime','Block','Trial',
                                            'M1','M2','Choice','RT','Outcome')))

## Concatenate DataFrames.
df = concat(df)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Postprocessing.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Handle missing data
## -------------------
## Missing choices and reaction times are set to NaNs. 
df.Choice = np.where(df.Choice==0, np.nan, np.where(df.Choice==1, df.M1, df.M2))
df.RT = np.where(df.Choice.isnull(), np.nan, df.RT)

## Re-sort by trial.
df = df.sort_values(['Datetime','Block','Trial'])    

## Insert unique subject index.
_, subj_ix = np.unique(df.Datetime, return_inverse=True)
df.insert(0, 'Subject', subj_ix+1)

## Save.
df.to_csv('data/moodRL_data.csv', index=False)
print('Done.')

import os
import numpy as np
from datetime import datetime
from pandas import DataFrame, concat
from scipy.io import loadmat

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Main loop.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

df = []
for cohort in ['behav','fmri']:
    
    ## Locate files.
    raw_dir = os.path.join('raw', cohort)
    files = os.listdir(raw_dir)

    ## Iterate over files.
    for f in files:
    
        ## Load and extract data.
        mat = loadmat(os.path.join(raw_dir, f))
        data = mat['Data'].squeeze()

        ## Extract subject/datetime info.
        subject, date, time = f.replace('.mat','').split('_')
        date = '-'.join(['%0.2d' %int(s) for s in date.split('-')])
        time = '-'.join(['%0.2d' %int(s) for s in time.split('-')])
        dt = datetime.strptime('%s %s' %(date,time), '%Y-%m-%d %H-%M-%S')

        ## Iterate over blocks.
        for blockno, block in enumerate(data[:-1]):

            ## Separate data categories
            ## ------------------------
            ## - outcome: binary array indicating win (1) or loss (0).
            ## - onsets: list of arrays with task event onsets.
            ## - choice: integer array indicating machine chosen.
            ## - ratings: list of arrays with mood/probability ratings.
            ## - info: list of arrays of block information.
            outcome, onsets, choice, ratings, info = block

            ## Organize ratings
            ## ----------------
            ## Mood and probability ratings are queried on the (7, 21, 35) and 
            ## (14, 28, 42) trial respectively. Probability judgments are 
            ## re-sorted by objective probability of reward in ascending order. 
            _, _, identities = [arr.squeeze() for arr in info.squeeze().tolist()]
            _, _, moods, probabilities = ratings.squeeze().tolist()
            probabilities = probabilities[:,identities-1]

            ## Merge ratings.
            trials = [7, 21, 35] + [14, 28, 42] * 3
            ratings = np.concatenate([moods.squeeze(), probabilities.flatten(order='F')])
            variables = np.repeat(np.arange(1+3*blockno,4+3*blockno), 3)
            variables = np.concatenate([np.repeat('Mood',3), variables])

            ## Assemble DataFrame
            ## ------------------
            ## Stores the following information: subject ID, datetime, 
            ## block number, rating type, trial number, rating.
            d = dict(Cohort = np.repeat(cohort, ratings.size),
                     Datetime = np.repeat(dt, ratings.size),
                     Block = np.repeat(blockno+1, ratings.size),
                     Trial = trials,
                     Variable = variables, 
                     Rating = ratings)
            df.append(DataFrame(d, columns=('Cohort','Datetime','Block','Trial',
                                            'Variable','Rating')))

        ## Extract mood ratings collected at beginning of blocks 1 & 2.
        mood_ix = np.argmax(np.in1d(mat['Rt'].dtype.names, 'mood'))
        moods = mat['Rt'].squeeze().tolist()[mood_ix].squeeze()

        ## Assemble DataFrame.
        d = dict(Cohort = np.repeat(cohort, 2),
                 Datetime = np.repeat(dt, 2),
                 Block = np.arange(2) + 1,
                 Trial = np.zeros(2, dtype=int),
                 Variable = np.repeat('Mood',2), 
                 Rating = moods)
        df.append(DataFrame(d, columns=('Cohort','Datetime','Block','Trial','Variable','Rating')))
        
## Concatenate DataFrames.
df = concat(df)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Postprocessing.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Re-sort by trial.
df = df.sort_values(['Datetime','Block','Trial'])

## Insert unique subject index.
_, subj_ix = np.unique(df.Datetime, return_inverse=True)
df.insert(0, 'Subject', subj_ix+1)

## Save.
df.to_csv('data/moodRL_ratings.csv', index=False)
print('Done.')

import os
import numpy as np
from datetime import datetime
from pandas import DataFrame, Series, read_csv, to_datetime
from scipy.io import loadmat

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## I/O parameters.
raw_dir = 'raw'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Main loop.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Locate files.
files = sorted(os.listdir(raw_dir)) 

df = []
for cohort in ['behav','fmri']:
    
    ## Locate files.
    raw_dir = os.path.join('raw', cohort)
    files = os.listdir(raw_dir)

    ## Iterate over files.
    for f in files:
    
        ## Load data.
        mat = loadmat(os.path.join(raw_dir, f))

        ## Extract subject/datetime info.
        subject, date, time = f.replace('.mat','').split('_')
        date = '-'.join(['%0.2d' %int(s) for s in date.split('-')])
        time = '-'.join(['%0.2d' %int(s) for s in time.split('-')])
        dt = datetime.strptime('%s %s' %(date,time), '%Y-%m-%d %H-%M-%S')

        ## Initialize Series object
        ## ------------------------
        ## In this first step, a Pandas Series object is initialized with the 
        ## following information: subject ID, datetime, eyetracking recorded, 
        ## fMRI recorded, Wheel of Fortune outcome.
        eyetrack = int(mat.get('eyetrack', 0))
        WoF = float(mat['WOF'].squeeze().tolist()[0])
        series = Series([cohort,dt,eyetrack,WoF], 
                        index=('Cohort','Datetime','Eyetrack','WoF'))

        ## Extract and store survey data
        ## -----------------------------
        ## In this step, the survey information is extracted from the Matlab
        ## object and stored in the Series. Information is subdivided by 
        ## survey scale. 
        for data, survey in zip(mat['Rt'].squeeze().tolist(), mat['Rt'].dtype.names):

            ## Extract subscale names/data. Skip mood questions (stored elsewhere).
            scales = data.dtype.names
            data = np.array(data.squeeze().tolist()).squeeze()
            if survey == 'mood': continue 

            ## Iteratively store.
            for datum, scale in zip(data, scales):
                series['%s_%s' %(survey,scale)] = datum

        ## Append series object.
        df.append(series)
    
## Concatenate Series objects.
df = DataFrame(df)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Postprocessing.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Re-sort.
df = df.sort_values('Datetime')

## Insert unique subject index.
_, subj_ix = np.unique(df.Datetime, return_inverse=True)
df.insert(0, 'Subject', subj_ix+1)

## Calculate earnings
## ------------------
## Total earnings are calculated based on the summed total outcomes
## of the reinforcement learning task (Blocks 1-3) and machine 
## discrimination task (Block 4), and the outcome on the Wheel of
## Fortune task. The former are calculated from the RL data spreadsheet.
data = read_csv('data/moodRL_data.csv')
data['Datetime'] = to_datetime(data['Datetime'])
outcomes = data.groupby(['Subject','Datetime']).Outcome.sum()

df['Outcomes'] = outcomes.as_matrix()
df['Earnings'] = df.WoF + df.Outcomes

## Reorganize columns.
df = df[['Subject', 'Cohort', 'Datetime', 'Eyetrack', 'Outcomes', 'WoF', 'Earnings',
         'BISBAS_basd', 'BISBAS_basf', 'BISBAS_basr', 'BISBAS_bis', 'IPIP_agr', 
         'IPIP_con', 'IPIP_ext', 'IPIP_hps', 'IPIP_neu', 'IPIP_opn', 'PANAX_neg', 'PANAX_pos']]

## Save.
df.to_csv('data/moodRL_metadata.csv', index=False)
print('Done.')

