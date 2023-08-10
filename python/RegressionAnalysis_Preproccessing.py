get_ipython().magic('pylab inline')

import pandas as pd

def hot1_encoding(df, index):
    s= df[index]
    m_value_counts= s.value_counts()
    for k,v in m_value_counts.iteritems():
        uv_mask= (df[index] == k).astype(float)
        new_field= "Field_sourceCol_{}_value_{}".format(index, k)
        df[new_field]= uv_mask
    
    del df[index]

from datetime import datetime

def extract_day_of_week(s):
    d= datetime.strptime(s, '%m-%d-%Y')
    return d.weekday()

def get_reconstruction_from_projection(eigenvectors, Mean, m_projection):
    Eig= np.matrix(eigenvectors)
    rec = np.array(m_projection * Eig.transpose() + Mean)
    #rec= Eig*m_projection.transpose()+Mean[:,np.newaxis]
    return np.ravel(rec)

m_fwy_meta_df= pd.read_json('./data/regression/station_meta.json', typ='frame', orient='records')
m_fwy_meta_df.rename(columns={'station': 'S_ID', 'district' : 'DISTRICT_ID', 'latitude' : 'LAT', 'longitude' : 'LON', 'zip' : 'ZIP'}, inplace=True)
m_fwy_meta_df.drop(labels=['direction', 'freeway', 'name', 'urban'], axis=1, inplace=True)
m_fwy_meta_df

#
# Memory Error occurs when attempting to run for all years and partitions, adjust list parameters as necessary
#
# p1_list= ['wkday', 'wkend']
# p2_list= ['weekday', 'weekend']
p1_list= ['wkend']
p2_list= ['weekend']
partitions= zip(p1_list, p2_list)
years= [2008, 2009, 2010, 2011, 2013, 2014, 2015]
#years= [2015]
#
hot1_columns= ['NUM_LANES', 'FWY_NUM', 'FWY_DIR', 'DAY_OF_WEEK', 'DISTRICT_ID']
for pentry in partitions:
    for y in years:
        p1= pentry[0]
        p2= pentry[1]
        #
        print('Processing {}, {}, {}'.format(p1, p2, y))
        a_df= pd.read_csv('./data/regression/trim_{}_{}.csv'.format(y, p1), header=0)
        c_df= pd.merge(a_df, m_fwy_meta_df, on='S_ID')
        #
        base_mean_path= './data/{}/total_flow_{}_mean_vector.pivot_{}_grouping_pca_tmp.csv'
        base_eigs_path= './data/{}/total_flow_{}_eigenvectors.pivot_{}_grouping_pca_tmp.csv'
        mean= pd.read_csv(base_mean_path.format(p2, p2, y), header=None).values[0]
        eigs= pd.read_csv(base_eigs_path.format(p2, p2, y), header=None).values  # eigenvectors per row matrix (5 X 288)

        rows= c_df[['Flow_Coef_1', 'Flow_Coef_2', 'Flow_Coef_3', 'Flow_Coef_4', 'Flow_Coef_5']].values

        new_columns= np.zeros(len(rows))
        for i, row in enumerate(rows):
            rec= get_reconstruction_from_projection(eigs, mean, row)
            new_columns[i]= np.mean(rec)

        c_df['AGG_TOTAL_FLOW']= new_columns
        c_df.drop([
            'S_ID',
            'Flow_Coef_1',
            'Flow_Coef_2',
            'Flow_Coef_3', 
            'Flow_Coef_4', 
            'Flow_Coef_5',
            'CHP_DESC',
            'CHP_DURATION',
            'CC_CODE',
            'ZIP'
        ], axis=1, inplace=True)
        #
        c_df['CHP_INC']= c_df.CHP_INC.apply(lambda v: 1 if v == 'T' else 0)
        c_df['CHP_INC']= c_df.CHP_INC.astype(float)

        c_df['DATE']= c_df.DATE.apply(lambda s: extract_day_of_week(s))
        c_df.rename(columns={'DATE':'DAY_OF_WEEK'}, inplace=True)
        #
        for c in hot1_columns:
            hot1_encoding(c_df, c)
        c_df.to_csv('./data/regression/preprocessed_{}_{}.csv'.format(y, p1), index=False)
        #
        del c_df
        del a_df
        del rows

