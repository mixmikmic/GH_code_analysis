import pandas as pd

liwc = pd.read_csv('LIWC_port.txt',encoding='Latin5')

liwc.head()

def get_funct(row):
    if '[1]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_pronoun(row):
    if '[2]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_ppron(row):
    if '[3]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_i(row):
    if '[4]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_we(row):
    if '[5]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_you(row):
    if '[6]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_shehe(row):
    if '[7]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_they(row):
    if '[8]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_ipron(row):
    if '[9]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_article(row):
    if '[10]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_verb(row):
    if '[11]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_auxverb(row):
    if '[12]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_past(row):
    if '[13]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_present(row):
    if '[14]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_future(row):
    if '[15]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_adverb(row):
    if '[16]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_preps(row):
    if '[17]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_conj(row):
    if '[18]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_negate(row):
    if '[19]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_quant(row):
    if '[20]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_number(row):
    if '[21]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_swear(row):
    if '[22]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_social(row):
    if '[121]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_family(row):
    if '[122]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_friend(row):
    if '[123]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_humans(row):
    if '[124]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_affect(row):
    if '[125]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_posemo(row):
    if '[126]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_negemo(row):
    if '[127]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_anx(row):
    if '[128]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_anger(row):
    if '[129]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_sad(row):
    if '[130]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_cogmech(row):
    if '[131]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_insight(row):
    if '[132]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_cause(row):
    if '[133]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_discrep(row):
    if '[134]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_tentat(row):
    if '[135]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_certain(row):
    if '[136]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_inhib(row):
    if '[137]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_incl(row):
    if '[138]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_excl(row):
    if '[139]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_percept(row):
    if '[140]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_see(row):
    if '[141]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_hear(row):
    if '[142]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_feel(row):
    if '[143]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_bio(row):
    if '[146]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_body(row):
    if '[147]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_health(row):
    if '[148]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_sexual(row):
    if '[149]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_ingest(row):
    if '[150]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_relativ(row):
    if '[250]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_motion(row):
    if '[251]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_space(row):
    if '[252]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_time(row):
    if '[253]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_work(row):
    if '[354]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_achieve(row):
    if '[355]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_leisure(row):
    if '[356]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_home(row):
    if '[357]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_money(row):
    if '[358]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_relig(row):
    if '[359]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_death(row):
    if '[360]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_assent(row):
    if '[462]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_nonfl(row):
    if '[463]' in row['related_sentiments']:
         return 1
    else:
        return 0
def get_filler(row):
    if '[464]' in row['related_sentiments']:
         return 1
    else:
        return 0

# Now I will assign all the results I get to a respective category
liwc['funct'] = liwc.apply(get_funct,axis=1)
liwc['pronoun'] = liwc.apply(get_pronoun,axis=1)
liwc['ppron'] = liwc.apply(get_ppron,axis=1)
liwc['i'] = liwc.apply(get_i,axis=1)
liwc['we'] = liwc.apply(get_we,axis=1)
liwc['you'] = liwc.apply(get_you,axis=1)
liwc['shehe'] = liwc.apply(get_shehe,axis=1)
liwc['they'] = liwc.apply(get_they,axis=1)
liwc['ipron'] = liwc.apply(get_ipron,axis=1)
liwc['article'] = liwc.apply(get_article,axis=1)
liwc['verb'] = liwc.apply(get_verb,axis=1)
liwc['auxverb'] = liwc.apply(get_auxverb,axis=1)
liwc['past'] = liwc.apply(get_past,axis=1)
liwc['present'] = liwc.apply(get_present,axis=1)
liwc['future'] = liwc.apply(get_future,axis=1)
liwc['adverb'] = liwc.apply(get_adverb,axis=1)
liwc['preps'] = liwc.apply(get_preps,axis=1)
liwc['conj'] = liwc.apply(get_conj,axis=1)
liwc['negate'] = liwc.apply(get_negate,axis=1)
liwc['quant'] = liwc.apply(get_quant,axis=1)
liwc['number'] = liwc.apply(get_number,axis=1)
liwc['swear'] = liwc.apply(get_swear,axis=1)
liwc['social'] = liwc.apply(get_social,axis=1)
liwc['family'] = liwc.apply(get_family,axis=1)
liwc['friend'] = liwc.apply(get_friend,axis=1)
liwc['humans'] = liwc.apply(get_humans,axis=1)
liwc['affect'] = liwc.apply(get_affect,axis=1)
liwc['posemo'] = liwc.apply(get_posemo,axis=1)
liwc['negemo'] = liwc.apply(get_negemo,axis=1)
liwc['anx'] = liwc.apply(get_anx,axis=1)
liwc['anger'] = liwc.apply(get_anger,axis=1)
liwc['sad'] = liwc.apply(get_sad,axis=1)
liwc['cogmech'] = liwc.apply(get_cogmech,axis=1)
liwc['insight'] = liwc.apply(get_insight,axis=1)
liwc['cause'] = liwc.apply(get_cause,axis=1)
liwc['discrep'] = liwc.apply(get_discrep,axis=1)
liwc['tentat'] = liwc.apply(get_tentat,axis=1)
liwc['certain'] = liwc.apply(get_certain,axis=1)
liwc['inhib'] = liwc.apply(get_inhib,axis=1)
liwc['incl'] = liwc.apply(get_incl,axis=1)
liwc['excl'] = liwc.apply(get_excl,axis=1)
liwc['percept'] = liwc.apply(get_percept,axis=1)
liwc['see'] = liwc.apply(get_see,axis=1)
liwc['hear'] = liwc.apply(get_hear,axis=1)
liwc['feel'] = liwc.apply(get_feel,axis=1)
liwc['bio'] = liwc.apply(get_bio,axis=1)
liwc['body'] = liwc.apply(get_body,axis=1)
liwc['health'] = liwc.apply(get_health,axis=1)
liwc['sexual'] = liwc.apply(get_sexual,axis=1)
liwc['ingest'] = liwc.apply(get_ingest,axis=1)
liwc['relativ'] = liwc.apply(get_relativ,axis=1)
liwc['motion'] = liwc.apply(get_motion,axis=1)
liwc['space'] = liwc.apply(get_space,axis=1)
liwc['time'] = liwc.apply(get_time,axis=1)
liwc['work'] = liwc.apply(get_work,axis=1)
liwc['achieve'] = liwc.apply(get_achieve,axis=1)
liwc['leisure'] = liwc.apply(get_leisure,axis=1)
liwc['home'] = liwc.apply(get_home,axis=1)
liwc['money'] = liwc.apply(get_money,axis=1)
liwc['relig'] = liwc.apply(get_relig,axis=1)
liwc['death'] = liwc.apply(get_death,axis=1)
liwc['assent'] = liwc.apply(get_assent,axis=1)
liwc['nonfl'] = liwc.apply(get_nonfl,axis=1)
liwc['filler'] = liwc.apply(get_filler,axis=1)

# And I have a dataframe in the same format I want!
liwc.head()

# Finally, there are some * that mark a word that has multiple meanings.
liwc['word'] = liwc['word'].str.strip('*')

# Saving it to CSV
liwc.to_csv('liwc_formatted.csv',index=False)



