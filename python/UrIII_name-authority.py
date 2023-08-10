import pandas as pd
import re

with open('data/query_text_16_12_29-052645.txt', mode = 'r', encoding = 'utf8') as f:
    linelist = f.read().splitlines()

def clean(data, oldlist, new=""):
    for i in range(len(oldlist)):
        data = data.replace(oldlist[i],new)
    return data

remlist = ["`", "´", "/", "]", "[", "!", "?", "<", ">", "(", ")"]
lines = [clean(line, remlist) for line in linelist]

lines = [line for line in lines if not line[0].isdigit()]
lines = [line for line in lines if not "\t#" in line]

names = [word for line in lines for word in line.split() if word[0].isupper() or word.startswith('{d}')]
len(names)

names =[word for word in names if not word.isupper()]
len(names)

#names = [word for word in names if not word.endswith('{ki}')]
#names = [word for word in names if not '{ki}' in word]
#len(names)

names = [word for word in names if not '...' in word]
names = [word for word in names if not '-x' in word.lower() 
         and not 'x-' in word.lower() 
         and not '.x' in word.lower() 
         and not 'x.' in word.lower()
        and not '}x' in word.lower()
        and not 'x{' in word.lower()]
len(names)

total_names = len(names)
#keep the total number of names for statistics
names = set(names)
len(names)

names = sorted(names)
names

df = pd.DataFrame(names)

df.columns = ['Transliteration']

df['Normalized'] = df['Transliteration']

signs = {'c': 'š',
        'C': 'Š',
        'ty': 'ṭ',
        'TY': 'Ṭ',
        'sy': 'ṣ',
        'SY': 'Ṣ'}
for key in signs:
    df['Normalized'] = [word.replace(key, signs[key]) for word in df['Normalized']]

def signreplace(old, new, data):
    old_cap = old.capitalize()
    new_cap = new.capitalize()
    data = re.sub('\\b'+old_cap+'\\b', new_cap, data)
    data = re.sub('\\b'+old+'\\b', new, data)
    return data

bdtns_oracc = {'ag2': 'aŋ2',
               'balag': 'balaŋ', 
               'dagal': 'daŋal', 
               'dingir': 'diŋir',
               'eridu': 'eridug',
               'ga2': 'ŋa2', 
               'gar': 'ŋar',
               'geštin': 'ŋeštin',
               'gir2': 'ŋir2',
               'gir3': 'ŋiri3',
               'giri3': 'ŋiri3',
               'giš': 'ŋeš',
               'giškim': 'ŋiškim',
               'gišnimbar' : 'ŋešnimbar',
               'hun': 'huŋ',
               'kin': 'kiŋ2', 
               'nig2': 'niŋ2',
               'nigin': 'niŋin',
               'nigin2': 'niŋin2',
               'pisan': 'bisaŋ',
               'pirig': 'piriŋ',
               'sag': 'saŋ',
               'sanga': 'saŋŋa',
               'šeg3': 'šeŋ3', 
               'šeg6': 'šeŋ6',
               'umbisag': 'umbisaŋ',
               'uri2': 'urim2',
               'uri5': 'urim5',
               'uru': 'iri',
               
               'bara2' : 'barag',
               'du10' : 'dug3',
               'du11' : 'dug4',
               'gu4' : 'gud',
               'kala' : 'kalag',
               'ku3' : 'kug',
               'ku5': 'kud',
              # 'lu5' : 'lul',
               'sa6' : 'sag9',
               'ša6' : 'sag9',
               'za3' : 'zag'
           } 

for key in bdtns_oracc:
    df['Normalized'] = [signreplace(key, bdtns_oracc[key], word) for word in df['Normalized']]
#normalized = [signreplace(key, bdtns_oracc[key], word) for key in bdtns_oracc for word in df['Normalized']]

df['Normalized'] = [re.sub('{d}([a-zšŋ])', lambda match: '{d}'+'{}'.format(match.group(1).upper()), word) 
                    for word in df['Normalized']]

df

df['Normalized'] = [word[:-3] if word.endswith('-ta') else word for word in df['Normalized']]
df['Normalized'] = [word[:-4] if word.endswith('-še3') else word for word in df['Normalized']]
df['Normalized'] = [word[:-2] if word.endswith('a|e|i|u' + '-ke4') else word for word in df['Normalized']]
df['Normalized'] = [word[:-4] if word.endswith('-ke4') else word for word in df['Normalized']]
df['Normalized'] = [word[:-2]+'ŋu10' if word.endswith('-mu') else word for word in df['Normalized']]

df[10000:10050]

df['Normalized'] = [word + '-k' if word + '-k' in df.Normalized.values else word for word in df['Normalized']]

df[10000:10050]

df['Normalized'] = [signreplace('a-a', 'aya', word) for word in df['Normalized']]
df[:100]

remove = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ':']
df['Normalized'] = [clean(word, remove) if word[1:].islower() 
                    or (word.startswith('{d}') and word[4:].islower()) else word for word in df['Normalized']]

df['Normalized'] = [clean(word, remove) if ('{d}' in word[1:] 
              and word.split('{d}')[0][1:].islower() and word.split('{d}')[1][1:].islower()) else word 
              for word in df['Normalized']]
df[df['Normalized'].str.contains('{d}Dumu')]

df['Normalized'] = [clean(word, remove) if ('{d}' in word[1:] and word.startswith('{d}')
              and word.split('{d}')[1][1:].islower() and word.split('{d}')[2][1:].islower()) else word 
              for word in df['Normalized']]
df[df['Transliteration'].str.contains('Suen')]

gods_no_d = ['I-šum', 'E2-a', 'Er3-ra', 'A-šur5']
for god in gods_no_d:
    df['Normalized'] = [clean(word, remove) if (god in word[2:] 
              and re.split('-(?=' + god + '\\b)', word)[0][1:].islower() 
              and re.split('-(?=' + god + '\\b)', word)[1][1:].islower()) else word 
              for word in df['Normalized']]
df[df['Transliteration'].str.contains('E2-a')]

df['Normalized'] = [re.sub('\\B([a-z])\\1\\B', '\\1', word) for word in df['Normalized']]

vowel_combis = {"ae":"aʾe",
          "ai": "aʾi",
          "au": "aʾu",
          "ea": "eʾa",
          "ei": "eʾi",
          "eu": "eʾu",
          "ia": "iʾa",
          "ie": "iʾe",
          "iu": "iʾu",
          "ua": "uʾa",
          "ue": "uʾe",
          "ui": "uʾi"
         }


for key in vowel_combis:
    df['Normalized'] = [word.replace(key, vowel_combis[key]) for word in df['Normalized']]
df

df['Normalized'] = [word+'[]PN' for word in df['Normalized']]
# Settlement names
df['Normalized'] = [word[:-2]+'SN' if '{ki}' in word else word for word in df['Normalized']]

# Divine names
df['Normalized'] = [word[:-2]+'DN' if word.startswith('{d}') and not '{d}' in word[4:] 
                    else word for word in df['Normalized']]

# Royal names
Shulgi = ['Šulgira', 'Šulgi', '{d}Šulgira', '{d}Šulgi']
AmarSuen = ['Amar{d}Suʾenra', 'Amar{d}Suʾen', 'Amar{d}Suʾenka', '{d}Amar{d}Suʾenra', 
            '{d}Amar{d}Suʾen', '{d}Amar{d}Suʾenka']
ShuSuen = ['Šu{d}Suʾen', 'Šu{d}Suʾenra', 'Šu{d}Suʾenka',
          '{d}Šu{d}Suʾen', '{d}Šu{d}Suʾenra', '{d}Šu{d}Suʾenka']
IbbiSuen = ['Ibi{d}Suʾen', 'Ibi{d}Suʾenra', 'Ibi{d}Suʾenka',
          '{d}Ibi{d}Suʾen', '{d}Ibi{d}Suʾenra', '{d}Ibi{d}Suʾenka']
df['Normalized'] = ['Šulgir[]RN' if word[:-4] in Shulgi else word for word in df['Normalized']]
df['Normalized'] = ['AmarSuʾen[]RN' if word[:-4] in AmarSuen else word for word in df['Normalized']]
df['Normalized'] = ['ŠuSuʾen[]RN' if word[:-4] in ShuSuen else word for word in df['Normalized']]
df['Normalized'] = ['IbbiSuʾen[]RN' if word[:-4] in IbbiSuen else word for word in df['Normalized']]
df['Normalized'] = ['UrNammak[]RN' if 'Ur{d}Nama' in word else word for word in df['Normalized']]

df[df['Normalized'].str.contains('[]RN', regex=False)]

df['Normalized'] = [re.sub('{.+?}', '', word) if not '-' in word and not '.' in word 
                    else word for word in df['Normalized']]
df

numbers_index = {'0':'₀',
               '1': '₁',
               '2':'₂',
               '3':'₃',
               '4':'₄',
               '5':'₅',
               '6':'₆',
               '7':'₇',
               '8':'₈',
               '9':'₉',
                '-': '.'}

for key in numbers_index:
    df['Normalized'] = [word.replace(key, numbers_index[key]) for word in df['Normalized']]
df

not_norm_df = df[df['Normalized'].str.contains('.', regex=False)]
norm_df = df[~df['Normalized'].str.contains('.', regex=False)]
not_normalized = len(not_norm_df)
norm = len(norm_df)
norm_set = len(set(norm_df['Normalized']))
names_forms = len(df)
print('Name Instances ' + str(total_names))
print('Name forms: ' + str(names_forms))
print('Name forms Normalized: ' + str(norm) + "; representing " + str(norm_set) + " different names.")
print('Name forms not normalized: ' + str(not_normalized))

mica = df[df['Transliteration'].str.contains('mi-ca')]
Utamicaram = mica[mica['Transliteration'].str[0]=='U']
Utamicaram

df.to_csv('output/UrIII-Name_au.csv', sep = '\t', encoding='utf-16')



