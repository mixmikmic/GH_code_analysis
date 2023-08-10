import pickle
import pandas as pd
exp = pickle.load(open('new_lime_explanations_dict.p','rb'))
lowest_indices = pickle.load(open('lowest_indices.p', 'rb'))
highest_indices = pickle.load(open('highest_indices.p', 'rb'))
meta = pd.read_csv('ffc_variable_types.csv')

meta.index = meta['variable']
del meta['variable']

exp

explanations = {}
for k,v in exp.items():
    user_exp = {}
    for x in v:
        user_exp[x[0]] = x[1]
    explanations[k] = user_exp

explanations[15]

meta.loc['f3d3a_5']

df = pd.DataFrame.from_dict(explanations, orient='index')

df.shape

df.head()

df_low = df.loc[[x[0] for x in lowest_indices]]
df_high = df.loc[[x[0] for x in highest_indices]]
df_low = df_low.dropna(axis=1, how='all') # dropping empty columns
df_high = df_high.dropna(axis=1, how='all')

def extract_variable_name(s):
    """
    This function parses the column names in the explanations to extract the variable name
    from the FF survey.
    
    I have left in the comments to illustrate how the algorithm is working."""
    components = s.split()
    print(s)
    try: 
        float(components[0]) # if first component can be case to a float then var name in 2nd
        print('First component is a float')
        var = components[2]
        print('Name is in ', var)
    except ValueError:
        var = components[0]
        print('Name is in ', var)
        
    if '_' in var:
        subcomponents = var.split('_')
        if var.count('_') == 1:
            # if substring after the _ can't be cast to float then it is part of the name
            try:
                float(subcomponents[1])
                varname = subcomponents[0]
            except ValueError:
                varname = var
        elif var.count('_') > 1:
            print("More than one underscore in ", var)
            varname = subcomponents[0]+'_'+subcomponents[1]
            print("Variable name is ", varname)
            
    else:
        varname = var
    print(varname)
    return varname 

explanations_2 = {}
for k,v in explanations.items():
    user_exp = {}
    for x, v in v.items():
        var = extract_variable_name(x)
        user_exp[var] = v
    explanations_2[k] = user_exp
    
df_names = pd.DataFrame.from_dict(explanations_2, orient='index')

df_names.shape

low = df_names.loc[[x[0] for x in lowest_indices]]
high = df_names.loc[[x[0] for x in highest_indices]]

def get_variable_freqs(df):
    missingness = pd.DataFrame(df.isnull().sum())
    freq_dict = {}
    for r in missingness.iterrows():
        freq_dict[r[0]] = df.shape[0] - r[1][0]
    return freq_dict

low_freqs = get_variable_freqs(low)

high_freqs = get_variable_freqs(high)

for i, j in sorted(low_freqs.items(), key=lambda x: x[1], reverse=True):
    print(i, j, meta.loc[i]['label'], [x for x in list(df_low.columns) if i in x])

for i, j in sorted(high_freqs.items(), key=lambda x: x[1], reverse=True):
    print(i, j, meta.loc[i]['label'],  [x for x in list(df_high.columns) if i in x])

