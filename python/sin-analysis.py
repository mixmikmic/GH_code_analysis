import pandas as pd
import re

with open('../data/eLibrary_Schedule_Contracts.csv', encoding='ascii', errors='ignore') as f:
    rows = pd.read_csv(f)

rows.drop_duplicates(subset=['SPECIAL_ITEM_NUMBER'], inplace=True)

SINS = {}

for i in rows.index:
    sin = rows['SPECIAL_ITEM_NUMBER'][i]
    desc = rows['SPECIALITEM_NUMBER_DESCRIPTION'][i]
    SINS[sin] = desc

# From http://www.gsa.gov/portal/content/245439
SINS['100 01'] = 'Introduction of New Services'
SINS['100 03'] = 'Ancillary Supplies and/or Services'

SIN_DESCRIPTIONS = pd.Series(SINS)
SIN_DESCRIPTIONS.to_frame(name='Description')

prices = pd.read_csv('../data/hourly_prices.csv', index_col=False, thousands=',')

def parse_sins(prices):
    idx = {}
    charset = build_charset(SINS.keys())
    sins = SINS.keys()
    sin_map = {}
    for sin in sins:
        sin_map[tuple(sin.split(' '))] = True
    charset_re = re.compile('[' + build_charset(sins) + ']+')
    prefixes = get_sin_prefixes(SINS.keys())
    prefix_map = {}
    for prefix in prefixes:
        prefix_map[prefix] = True

    # TODO: Some SINs in our data are suffixed with strange characters
    # that don't have entries in our SIN descriptions table. However,
    # the version *without* the suffix is in our table, so we'll
    # strip out the suffix if needed. But we need to figure out what
    # those suffixes mean and whether what we're doing is OK.
    weird_suffixes = ['R', 'RC']

    for i in prices['SIN NUMBER'].index:
        val = str(prices['SIN NUMBER'][i])
        parts = charset_re.findall(val)
        last_prefix = None
        sins = []
        for part in parts:
            if last_prefix is not None:
                sin = (last_prefix, part)
                for weird_suffix in weird_suffixes:
                    if sin not in sin_map and part.endswith(weird_suffix):
                        sin = (last_prefix, part[:-len(weird_suffix)])
                        continue
                if sin in sin_map:
                    sins.append(' '.join(sin))
                    continue
            if part in prefix_map:
                last_prefix = part
            elif part.startswith('C') and part[1:] in prefix_map:
                # TODO: Some SINs in our data are prefixed with the letter
                # 'C' and aren't in our SIN descriptions table, but their
                # variant without the 'C' *is* in our table. We're going
                # to strip out the prefix for now, but we should figure
                # out what it means and whether what we're doing is OK.
                last_prefix = part[1:]
        if sins:
            idx[i] = list(set(sins))
    return pd.Series(idx)

def build_charset(vocab):
    charset = []
    for word in vocab:
        charset.extend(list(word))
    charset = set(charset)
    return ''.join(charset.difference([' ', '-']))

def get_sin_prefixes(sins):
    return list(set([sin.split(' ')[0] for sin in sins]))

sins = pd.DataFrame(index=prices.index)
sins['Raw SIN'] = prices['SIN NUMBER'].fillna('')
sins['Parsed SINs'] = parse_sins(prices)
sins['Labor Category'] = prices['Labor Category']

sins.head(5)

sins[sins['Parsed SINs'].isnull()].head(5)

sins[sins['Raw SIN'].str.contains('thru')].head(2)

def get_union(series):
    union = set()

    series[series.notnull()].apply(lambda x: union.update(x))
    return list(union)

priced_sins = pd.DataFrame(index=get_union(sins['Parsed SINs']))
priced_sins['Description'] = SIN_DESCRIPTIONS

def get_item_counts(series):
    counts = {}

    def add_counts(items):
        for item in items:
            counts[item] = counts.get(item, 0) + 1

    series[series.notnull()].apply(add_counts)
    return counts

priced_sins['Count'] = pd.Series(get_item_counts(sins['Parsed SINs']))

get_ipython().magic('matplotlib inline')

priced_sins.sort_values(by='Count', ascending=False)

bayesian_sin_words = {}
bayesian_words = {'_total': 0}

for sin in priced_sins.index:
    bayesian_sin_words[sin] = {'_total': 0}

for i in sins[sins['Parsed SINs'].notnull()].index:
    parsed_sins = sins['Parsed SINs'][i]
    # TODO: Consider using a stemmer here?
    words = sins['Labor Category'][i].lower().split()
    for word in words:
        if not word:
            continue
        bayesian_words[word] = bayesian_words.get(word, 0) + 1
        bayesian_words['_total'] += 1
        for sin in parsed_sins:
            bayesian_sin_words[sin][word] = bayesian_sin_words[sin].get(word, 0) + 1
            bayesian_sin_words[sin]['_total'] += 1

def get_word_probability(word, sin):
    '''Given a word, what is the probability it is associated with the given SIN?'''
    
    prob_of_word_given_sin = float(bayesian_sin_words[sin].get(word, 0)) / bayesian_sin_words[sin]['_total']
    prob_of_sin = float(priced_sins['Count'][sin]) / prices.shape[0]
    prob_of_word = float(bayesian_words.get(word, 0)) / bayesian_words['_total']
    if prob_of_word == 0:
        return 0
    return prob_of_word_given_sin * prob_of_sin / prob_of_word

def get_probable_sins(labor_category):
    probs = {}
    words = labor_category.lower().split()
    for sin in priced_sins.index:
        prob = 1
        for word in words:
            prob *= get_word_probability(word, sin)
        probs[sin] = prob
    df = pd.DataFrame(index=priced_sins.index)
    df['Description'] = SIN_DESCRIPTIONS
    df['Probability'] = pd.Series(probs)
    return df.sort_values(by='Probability', ascending=False)

get_probable_sins('creative director').head(10)



