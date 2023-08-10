import pandas as pd
from bs4 import BeautifulSoup
import requests
import pickle

if True:
    # Get MPs from all recent elections
    elections = [
        "1970-06-18", "1974-02-28", "1974-10-10", "1979-05-03", "1983-06-09",
        "1987-06-11", "1992-04-09", "1997-05-01", "2001-06-07", "2005-05-05",
        "2010-05-06", "2015-06-07", "2017-05-01", "2017-06-15"
    ]
    mp_dataframes = [ pd.read_csv("https://www.theyworkforyou.com/mps/?f=csv&date=" + election) for election in elections ]
    # Group MPs by ID to remove duplicate entries
    mps = pd.concat(mp_dataframes).drop_duplicates(
        subset="Person ID", keep="last").sort_values("Person ID")

if True:
    # Get list of women mps (from women-in-parliament R project)
    women_mps = pd.read_csv("mps_over_time.csv")
    # Match MPs by name to identify women in the list
    mps["full_name"] = mps["First name"].fillna("") + " " + mps["Last name"].fillna("")
    mps["clean_name"] = mps["full_name"].str.lower().replace(regex=True, to_replace=r'\W',value=r'')
    mps["is_female"] = mps["clean_name"].map(lambda x: x in women_mps["clean_name"].tolist())

if True:
    # Join women MP dataframe with mps dataframe to get party of women MPs
    mps = mps.join(women_mps[women_mps.party.notnull()][["clean_name", "party"]]
                   .drop_duplicates(keep="last")
                   .set_index("clean_name"),
                   on="clean_name").rename(columns={"party": "party_women"})

if True:
    # Replace party abbreviations with full names
    party_abbr = {
        "Lab": "Labour",
        "LD": "Liberal Democrat",
        "Con": "Conservative",
        "SNP": "Scottish National Party",
        "PC": "Plaid Cymru",
        "UU": "UUP",
        "Alliance": "Alliance",
        "SF": "Sinn Féin",
        "DU": "DUP",
        "SDLP": "Social Democratic and Labour Party",
        "SDP": "SDP",
        "Green": "Green",
        "Ind. Unity": "Independent"
    }

    mps.party_women = mps.party_women.apply(party_abbr.get)

import pandas as pd

def get_full_name(x, note="Main"):
    """Return MP names within JSON structure"""
    # Index of MP's main name
    try:
        i = list(map(lambda name: name["note"], x)).index(note)
    except ValueError:
        i = 0
    try:
        given_name = x[i]["given_name"]
    except KeyError:
        given_name = ""
    try:
        last_name = x[i]["family_name"]
    except KeyError:
        try:
            last_name = x[i]["surname"]
        except KeyError:
            last_name = ""
    try:
        return x[i]["name"]
    except KeyError:
        return given_name + " " + last_name

mps_mysoc = requests.get("https://github.com/mysociety/parlparse/blob/master/members/people.json?raw=true").json()
mp_mysoc = pd.DataFrame(mps_mysoc["persons"])[["other_names", "id", "identifiers"]].dropna()

mp_mysoc["full_name"] = mp_mysoc["other_names"].apply(get_full_name)
mp_mysoc["clean_name"] = mp_mysoc["full_name"].str.lower().replace(regex=True, to_replace=r'\W',value=r'')

mp_mysoc["id"] = mp_mysoc["id"].apply(lambda x: x.split("/")[-1])

def clean_name(name_series, strip_whitespace=True, lower=True):
    """Clean MP names in a series by removing honorifics, any non-alphabet characters and anything inside brackets"""
    import re
    
    honorifics = r'(Mr|Mrs|Ms|Miss|Advocate|Ambassador|Baron|Baroness|Brigadier|Canon|Captain|Chancellor|Chief|Col|Comdr|Commodore|Councillor|Count|Countess|Dame|Dr|Duke of|Earl|Earl of|Father|General|Group Captain|H R H the Duchess of|H R H the Duke of|H R H The Princess|HE Mr|HE Senora|HE The French Ambassador M|His Highness|His Hon|His Hon Judge|Hon|Hon Ambassador|Hon Dr|Hon Lady|Hon Mrs|HRH|HRH Sultan Shah|HRH The|HRH The Prince|HRH The Princess|HSH Princess|HSH The Prince|Judge|King|Lady|Lord|Lord and Lady|Lord Justice|Lt Cdr|Lt Col|Madam|Madame|Maj|Maj Gen|Major|Marchesa|Marchese|Marchioness|Marchioness of|Marquess|Marquess of|Marquis|Marquise|Master|Mr and Mrs|Mr and The Hon Mrs|President|Prince|Princess|Princessin|Prof|Prof Emeritus|Prof Dame|Professor|Queen|Rabbi|Representative|Rev Canon|Rev Dr|Rev Mgr|Rev Preb|Reverend|Reverend Father|Right Rev|Rt Hon|Rt Hon Baroness|Rt Hon Lord|Rt Hon Sir|Rt Hon The Earl|Rt Hon Viscount|Senator|Sir|Sister|Sultan|The Baroness|The Countess|The Countess of|The Dowager Marchioness of|The Duchess|The Duchess of|The Duke of|The Earl of|The Hon|The Hon Mr|The Hon Mrs|The Hon Ms|The Hon Sir|The Lady|The Lord|The Marchioness of|The Princess|The Reverend|The Rt Hon|The Rt Hon Lord|The Rt Hon Sir|The Rt Hon The Lord|The Rt Hon the Viscount|The Rt Hon Viscount|The Venerable|The Very Rev Dr|Very Reverend|Viscondessa|Viscount|Viscount and Viscountess|Viscountess|W Baron|W/Cdr)'
    h = re.compile(honorifics.replace("|", r" \b|\b"))
    
    name_series = name_series        .str.replace(h, "")        .replace(regex=True, to_replace=r' (CH|DBE|CBE|OBE|MBE|TD|QC)$', value=r'')        .replace(regex=True, to_replace=r'\(.*\)', value="")        .str.replace(",", "")
    
    if lower:
        name_series = name_series.str.lower()
    if strip_whitespace:
        name_series = name_series.replace(regex=True, to_replace=r'\W',value=r'')
    
    return name_series

import fuzzywuzzy.process

a = pd.read_csv("mps_over_time.csv").copy()
a["clean_name"] = clean_name(a["name"])

a = a.join(mp_mysoc.set_index("clean_name")[["id"]], on="clean_name")
mp_mysoc["clean_name_"] = mp_mysoc["other_names"].apply(lambda x: get_full_name(x, note="Alternate"))    .str.lower()    .replace(regex=True, to_replace=r'\W',value=r'')

a = a.join(mp_mysoc.set_index("clean_name_")[["id"]], on="clean_name", rsuffix="_")

a["id"] = a["id"].fillna(a["id_"])

a[a.name.str.contains("KEEN")]

mp_possibilities = list(filter(lambda x: len(x.split()) > 1, mp_mysoc[mp_mysoc["id"].isin(a[a["id"].notnull()]["id"].tolist()) == False].set_index("id")["full_name"].tolist()))

a.loc[a["id"].isnull(), "matches"] = clean_name(a.loc[a["id"].isnull()]["name"], strip_whitespace=False)    .apply(lambda x: fuzzywuzzy.process.extractBests(x, 
                                                     mp_possibilities,
                                                     scorer=fuzzywuzzy.fuzz.token_set_ratio,
                                                     score_cutoff=80))

def reverse_match(mp_name):
    if len(mp_name) > 0:
        return mp_mysoc[mp_mysoc["full_name"] == mp_name[0][0]].id.iloc[0]

a.loc[a["id"].isnull(), "id__"] = a.loc[a["id"].isnull(), "matches"].apply(reverse_match)

# Print out list of matches
print(a.loc[a["id"].isnull() & a["matches"].notnull()][["name", "id__"]].to_csv(index=False))

# Make some manual edits to the data for MPs that didn't match correctly
edits = """name,id__
"Katharine, Duchess of ATHOLL, DBE",16518
"Gwendolen, Countess of IVEAGH, CBE",17911
Lady Lucy NOEL-BUXTON,18613
Miss Marjorie GRAVES,21164
"Frances, Viscountess DAVIDSON",22505
Mrs Beatrice WRIGHT,19657
"Lady Violet APSLEY,",16489
"Priscilla, Lady TWEEDSMUIR",19376
Mrs Helene HAYMAN,13495
Rt Hon Dame Margaret BECKETT,10031
Mrs Helen BRINTON,10109
Celia BARLOW,11644
Julia GOLDSWORTHY,11581
Sarah McCARTHY-FRY,11766
Anne SNELGROVE,11866
Kitty USSHER,11464
Lynda WALTHO,11845
Ms Louise BAGSHAWE,24833
"""

from io import StringIO
edits = pd.read_csv(StringIO(edits)).set_index("name")
a = a.join(edits, on="name", rsuffix="_")
a["id"] = a["id___"].fillna(a.id__).fillna(a.id_).fillna(a.id)
#a = a.drop(["id_", "matches", "id__", "id___"], axis=1)

# Change type to integer
a["id"] = a["id"].astype(int)

# Save to csv file
a[["id", "name", "constituency", "term_start", "term_end", "party", "byelection", "notes", "clean_name", "stream"]].to_csv("women_mps.csv", index=False)

if True:
    # Get json list of all MPs catalogued in the mysociety database
    mps_mysociety = requests.get("https://github.com/mysociety/parlparse/blob/master/members/people.json?raw=true").json()

    # Turn into pandas dataframes
    party_id_crossmatch = pd.DataFrame(mps_mysociety["organizations"])[["id", "name"]]
    mp_party_crossmatch = pd.DataFrame(mps_mysociety["memberships"]).dropna(subset=["on_behalf_of_id"])[["person_id", "on_behalf_of_id"]].drop_duplicates()

    # Match person_id with party names
    mp_party = mp_party_crossmatch.join(party_id_crossmatch.set_index("id"), on="on_behalf_of_id").groupby("person_id").first()[["name"]]

    # Match mps in our list with mps in publicwhip list to get party
    mps = mps.assign(mysoc=mps["Person ID"]                .apply(lambda x: "uk.org.publicwhip/person/" + str(x)))                .join(mp_party, on="mysoc", how="left")                .drop("mysoc", axis=1)                .rename(columns={"name":"party_mysoc"})

if True:
    import requests
    import pandas as pd
    # Get all mps that exist in wikidata.org
    wikidata_query = '''SELECT ?mp ?mpLabel ?dob ?dod ?party ?partyLabel ?hansard ?genderLabel WHERE {
      ?mp p:P39/ps:P39/wdt:P279* wd:Q16707842; # Q16707842 = member of UK parliament
      wdt:P102 ?party. # P102 = belonging to a political party
      OPTIONAL {?mp wdt:P569 ?dob} . #P569 = date of birth
      OPTIONAL {?mp wdt:P570 ?dod} . #P570 = date of death
      OPTIONAL {?mp wdt:P2015 ?hansard} . #P2015 = hansard id
      OPTIONAL {?mp wdt:P21 ?gender} . #21 = gender

      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }'''

    wikidata_data = requests.get('https://query.wikidata.org/bigdata/namespace/wdq/sparql',
                                 params={'query': wikidata_query, 'format': 'json'}).json()

    def extract_optional(row, key):
        try:
            return row[key]["value"]
        except KeyError:
            return None

    # Convert the results to a pandas dataframe
    wikidata_mps = pd.DataFrame([ {"mp":mp["mpLabel"]["value"], "mp_id":mp["mp"]["value"],
                                   "gender":extract_optional(mp, "genderLabel"),
                                   "dob":extract_optional(mp, "dob"), "dod":extract_optional(mp, "dod"),
                                   "party": mp["partyLabel"]["value"], "party_id": mp["party"]["value"],
                                  "hansard": extract_optional(mp, "hansard")} for mp in wikidata_data["results"]["bindings"] ])

    # Format date of birth/death columns as datetime
    wikidata_mps.dob = pd.to_datetime(wikidata_mps.dob.fillna("").apply(lambda x: x[:-len("T00:00:00Z")]), errors="coerce")
    wikidata_mps.dod = pd.to_datetime(wikidata_mps.dod.fillna("").apply(lambda x: x[:-len("T00:00:00Z")]), errors="coerce")

    # Filter out any MPs that were dead by 1970 (they couldn't possibly appear in parliament speeches post 1970)
    all_wikidata_mps = wikidata_mps.copy()
    wikidata_mps = wikidata_mps[(wikidata_mps.dod > "1969") | wikidata_mps.dod.isnull()]

b = mp_mysoc[mp_mysoc.id.astype(int).isin(a["id"].tolist())]    .assign(clean_name=lambda x: [get_full_name(mp) for mp in x.other_names],
            clean_name_=lambda x: [get_full_name(mp, note="Alternate") for mp in x.other_names])\
    .assign(clean_name=lambda x: clean_name(x.clean_name, False), clean_name_=lambda x: clean_name(x.clean_name_, False))[["id", "clean_name", "clean_name_"]]

all_mps = all_wikidata_mps.assign(cleaned_name=lambda x: clean_name(x.mp, False))          .set_index("cleaned_name")[["gender", "mp_id"]].drop_duplicates()
    
c = b.join(all_mps[["mp_id"]], on="clean_name")    .join(all_mps[["mp_id"]], on="clean_name_", rsuffix="_")    .assign(mp_id = lambda x: x.mp_id.fillna(x.mp_id_))
    
c["id"] = c["id"].astype(int)

mp_possibilities = list(filter(lambda x: len(x.split()) > 1, all_mps.reset_index().query('gender=="female"')["cleaned_name"].tolist()))

c.loc[c.mp_id.isnull(), "matches"]  = c["clean_name"].apply(lambda x: fuzzywuzzy.process.extractBests(x, 
                                                     mp_possibilities,
                                                     scorer=fuzzywuzzy.fuzz.token_set_ratio,
                                                     score_cutoff=80))

mps = pd.read_hdf("list_of_mps.h5")

import requests
import pandas as pd

# Json data of MPs from the past twenty years which have both mysociety and wikidata ids
everypol = requests.get("https://cdn.rawgit.com/everypolitician/everypolitician-data/72d2213c9e3c8efeba1bf88ec2dcd50cd1bb0f4b/data/UK/Commons/ep-popolo-v1.0.json").json()
# Make a dataframe from json
everypol = pd.DataFrame([{"mysoc_id": int(list(filter(lambda identifier: identifier["scheme"] == "parlparse", mp["identifiers"]))[0]["identifier"].split("/")[-1]),
  "wikidata_id": list(filter(lambda identifier: identifier["scheme"] == "wikidata", mp["identifiers"]))[0]["identifier"]} for mp in everypol["persons"]])

everypol.head()

c.join(everypol.set_index("mysoc_id"), on="id")    .assign(wikidata_id=lambda x: "http://www.wikidata.org/entity/" + x.wikidata_id)

# Crossreference the mysociety ids we have to get wikidata ids for remaining MPs
d = c.join(everypol.set_index("mysoc_id"), on="id")    .assign(wikidata_id=lambda x: "http://www.wikidata.org/entity/" + x.wikidata_id)    .assign(wikidata_id = lambda x: x.wikidata_id.fillna(x.mp_id))

def reverse_match(mp_name):
    """For MPs that have a fuzzywuzzy match, return the MP's id by crossreferencing with the dataframe of all MPs"""
    if len(mp_name) > 0:
        return all_mps.loc[mp_name[0][0]].mp_id

# Let's manually add missing or incorrect matches using Google, etc
edits = """id,clean_name,matched_id
13495,helene ,http://www.wikidata.org/entity/Q24165
14155,barbara castle,http://www.wikidata.org/entity/Q2883936
14190,jennie adamson,http://www.wikidata.org/entity/Q6153138
14611,rosie barnes,http://www.wikidata.org/entity/Q7368880
16384,diana maddock,http://www.wikidata.org/entity/Q1208851
16489, apsley,http://www.wikidata.org/entity/Q7933220
16515,nancy astor,http://www.wikidata.org/entity/Q195013
16518, atholl,http://www.wikidata.org/entity/Q6376234
16533,alice bacon,http://www.wikidata.org/entity/Q4725734
16865,elaine burton,http://www.wikidata.org/entity/Q5353173
16922,lynda chalker,http://www.wikidata.org/entity/Q1878797
17125,florence dalton,http://www.wikidata.org/entity/Q7382939
17146,constance de markievicz,http://www.wikidata.org/entity/Q195768
17313,evelyn emmet,http://www.wikidata.org/entity/Q5416313
17430,janet fookes,http://www.wikidata.org/entity/Q222985
17841,florence horsbrugh,http://www.wikidata.org/entity/Q334280
17911, iveagh,http://www.wikidata.org/entity/Q1557837
17922,lena jeger,http://www.wikidata.org/entity/Q15229376
17968,thelma cazalet,http://www.wikidata.org/entity/Q7781055
17973,elaine kellett,http://www.wikidata.org/entity/Q4355733
18112,joan lestor,http://www.wikidata.org/entity/Q6205220
18369,bernadette devlin,http://www.wikidata.org/entity/Q2628635
18613, noel-buxton,http://www.wikidata.org/entity/Q6698437
18673,sally oppenheim,http://www.wikidata.org/entity/Q1533969
18769,mary pickford,http://www.wikidata.org/entity/Q6778777
18988,hilda runciman,http://www.wikidata.org/entity/Q5761464
19241,edith summerskill,http://www.wikidata.org/entity/Q665743
19283,vera terrington,http://www.wikidata.org/entity/Q7920843
19376, grant of monymusk,http://www.wikidata.org/entity/Q7245582
19657,beatrice rathbone,http://www.wikidata.org/entity/Q18922340
21022,arabella lawrence,http://www.wikidata.org/entity/Q7648103
21164,frances graves,http://www.wikidata.org/entity/Q6766268
21249,helen shaw,http://www.wikidata.org/entity/Q5701951
21408,mary hamilton,http://www.wikidata.org/entity/Q6778791
21748,doris fisher,http://www.wikidata.org/entity/Q3714285
21769,eirene white,http://www.wikidata.org/entity/Q5349851
21770,eleanor rathbone,http://www.wikidata.org/entity/Q333992
21905,irene ward,http://www.wikidata.org/entity/Q6069293
21937,jennie lee,http://www.wikidata.org/entity/Q1686926
21938,joan vickers,http://www.wikidata.org/entity/Q6205517
22487,barbara gould,http://www.wikidata.org/entity/Q4858727
22505,joan davidson,http://www.wikidata.org/entity/Q5478591
22525,mervyn pike,http://www.wikidata.org/entity/Q6820932
22552,renée short,http://www.wikidata.org/entity/Q7313013
25259,alison margaret ,http://www.wikidata.org/entity/Q2647251"""

from io import StringIO
edits = pd.read_csv(StringIO(edits))
# Change type to integer
edits["id"] = edits["id"].astype(int)
d = d.set_index("id")
d.loc[edits["id"].tolist(), "wikidata_id"] = edits["matched_id"].tolist()
d = d.reset_index()

# Write lookup table to csv
d[["id", "clean_name", "wikidata_id"]]	.assign(wikidata_id=lambda x: x["wikidata_id"].str.replace("http://www.wikidata.org/entity/", ""))    .to_csv("women_wikidata.csv",index=False)

def get_dob_from_hansard(mp_id):
    """Go to MP's hansard page and scrape DOB, DOD and constituencies served"""
    import requests
    from bs4 import BeautifulSoup
    
    hansard_request = requests.get("http://hansard.millbanksystems.com/people/" + mp_id)
    soup = BeautifulSoup(hansard_request.content, 'html.parser')
    
    # Find DOB and DOD by looking for vcard tag and taking the next line
    try:
        dates = soup.select("h1.vcard")[0].next_sibling.strip()
    except IndexError:
        dates = ""
    # Find constituencies served by MP
    # Return both in a dict
    return pd.Series({"dates":dates,
            "constituencies": [constituency.text for constituency in soup.select("li.constituency a")]})


if True:
    # Do it with multiple processes
    from multiprocessing import Pool
    pool = Pool(16)

    # Select MPs that do not have DOB or DOD but do have a hansard id
    mps_to_scrape = wikidata_mps[wikidata_mps.hansard.notnull()]
    # Run scrape function on all of them
    mps_scraped = pd.DataFrame(pool.map(get_dob_from_hansard, list(mps_to_scrape.hansard)))
    pool.close()
    pool.join()
    # Copy the index over so that the rows match
    mps_scraped.index = mps_to_scrape.index
    # Split the date text into DOB and DOD and assign to new columns
    mps_hansard = pd.concat([mps_to_scrape, mps_scraped                         .apply(lambda x: pd.Series(sum([str(x["dates"]).split("-"),
                                                         [x["constituencies"]]], [])), axis=1)\
                         .rename(columns={0:"hansard_dob", 1:"hansard_dod", 2:"hansard_constituencies"})],
                            axis=1)
    # Format dates as datetime
    mps_hansard["hansard_dob"] = pd.to_datetime(mps_hansard["hansard_dob"], errors="coerce") # If only year is recorded, it uses the 1st of January
    mps_hansard["hansard_dod"] = pd.to_datetime(mps_hansard["hansard_dod"], errors="coerce") # If only year is recorded, it uses the 1st of January

if True:
    # Bring hansard data back into wikidata dataframe
    wikidata_mps = wikidata_mps.join(mps_hansard[["hansard_dob", "hansard_dod", "hansard_constituencies"]])

    # Fill in empty DOB and DOD using hansard data
    wikidata_mps["dob"] = wikidata_mps["dob"].fillna(wikidata_mps["hansard_dob"])
    wikidata_mps["dod"] = wikidata_mps["dod"].fillna(wikidata_mps["hansard_dod"])

    # Create a year of birth column (can't use full date of birth because older MPs don't have accurate DOB or only have year of birth)
    wikidata_mps["yob"] = pd.to_numeric(wikidata_mps["dob"].dt.year.fillna(0).astype("int"))

    # Use both mp name and year of birth to group so that we don't accidentally match different MPs
    wikidata_mps["mp_yob"] = wikidata_mps["mp"] + "_" +wikidata_mps["yob"].astype(str)

def flatten_mps(mp_group):
    """Flatten a group of rows belonging to one MP into one row"""
    flattened_data = pd.Series()
    try:
        flattened_data["dob"] = pd.to_datetime(mp_group["dob"].dropna().iloc[0])
    except IndexError:
        flattened_data["dob"] = pd.NaT
    try:
        flattened_data["dod"] = pd.to_datetime(mp_group["dod"].dropna().iloc[0])
    except IndexError:
        flattened_data["dod"] = pd.NaT
    flattened_data["gender"] = mp_group["gender"].iloc[0]
    flattened_data["mp"] = mp_group["mp"].iloc[0]
    flattened_data["party"] = mp_group["party"].unique().tolist()
    flattened_data["mp_id"] = mp_group["mp_id"].unique().tolist()[0]
    try:
        # Collapse all lists of constituencies into one set of constituencies
        flattened_data["hansard_constituencies"] = set(sum(mp_group["hansard_constituencies"].dropna().tolist(), []))
    except TypeError:
        print(mp_group)
    
    return flattened_data

if True:
    # Filter by popular parties only
    wikidata_mps = wikidata_mps[wikidata_mps["party"].isin(
        ['Labour Party',
     'Conservative Party',
     'Plaid Cymru',
     'Liberal Democrats',
     'Liberal Party',
     'Ulster Unionist Party',
     'Scottish National Party',
     'Labour Co-operative',
     'Democratic Unionist Party',
     'Social Democratic Party',
     'Social Democratic and Labour Party',
     'UK Independence Party',
     'Green Party',
     'Sinn Féin',
     'Alliance Party of Northern Ireland',
     'Respect Party',
     'Co-operative Party'])]\
    .groupby("mp_yob").apply(flatten_mps) # Collapse all the duplicate rows into one per MP.
                                          # Constituencies are returned as sets, parties are returned as lists
    
    # Change them to datetime again...
    wikidata_mps["dob"] = pd.to_datetime(wikidata_mps["dob"])
    wikidata_mps["dod"] = pd.to_datetime(wikidata_mps["dod"])

if True:
    # Add some MP constituencies manually so that the matching goes smoothly
    wikidata_mps.query("mp_yob=='Angela Smith_1961'")["hansard_constituencies"].iloc[0].add("Penistone and Stocksbridge")
    wikidata_mps.query("mp=='John Foster'")["hansard_constituencies"].iloc[0].add("Northwich")
    wikidata_mps.query("mp=='Alan Brown'")["hansard_constituencies"].iloc[0].add("Kilmarnock and Loudoun")
    wikidata_mps.query("mp_yob=='Mike Wood_1976'")["hansard_constituencies"].iloc[0].add("Dudley South")
    wikidata_mps.query("mp=='Neil Carmichael'")["hansard_constituencies"].iloc[0].add("Stroud")
    wikidata_mps.query("mp=='Iain Stewart'")["hansard_constituencies"].iloc[0].add("Milton Keynes South")
    wikidata_mps.query("mp=='Donald Stewart'")["hansard_constituencies"].iloc[0].add("Na h-Eileanan an Iar")
    wikidata_mps.query("mp_yob=='Stewart McDonald_1986'")["hansard_constituencies"].iloc[0].add("Glasgow South")
    wikidata_mps.query("mp_yob=='Ian Paisley, Jr._1966'")["hansard_constituencies"].iloc[0].add("North Antrim")
    wikidata_mps.query("mp_yob=='Geoffrey Clifton-Brown_1953'")["hansard_constituencies"].iloc[0].add("The Cotswolds")
    mps.loc[mps["full_name"]=='Ian Paisley Jnr', "full_name"] = "Ian Paisley, Jr."

def match_mp(mp):
    """Use fuzzy string matching to match an MP by name.
    If there are multiple MPs with similar names, try to disambiguate using DOB and constituencies"""

    from fuzzywuzzy import fuzz, process
    
    matched_mps = process.extractBests(mp["full_name"], wikidata_mps["mp"],
                             scorer=fuzz.partial_token_sort_ratio, # computationally expensive scorer, but works well
                             score_cutoff=95)
    matched_mps_2 = process.extractBests(mp["full_name"], wikidata_mps["mp"],
                             scorer=fuzz.token_set_ratio, # alternate scorer which also works well
                             score_cutoff=95)
    # Combine both types of matches to get best results
    matched_mps.extend(matched_mps_2)
    # Extract mp_yob key and put in set
    matched_mps = {mp[2] for mp in matched_mps}
    # If there is only one fuzzy match, then assume we have our MP

    if len(matched_mps) == 1:
        # Get party from wikidata table
        party = wikidata_mps["party"].loc[list(matched_mps)[0]]
        # Get wikidata id from table
        mp_id = wikidata_mps["mp_id"].loc[list(matched_mps)[0]]
        return (party, mp_id, list(matched_mps)[0])
    elif len(matched_mps) > 1:
        # We have several matches so let's try to disambiguate by constituencies served
        # From the wikidata table, find these matched MP_yods, 
        matched_mp_yobs =  wikidata_mps.loc[matched_mps]
        # Then check if requested mp's constituency is in the list of constituencies of each match
        matched_mps_in_wiki = matched_mp_yobs["hansard_constituencies"]            .apply(lambda x: (process.extractOne(mp["Constituency"], x, scorer=fuzz.token_set_ratio)  or [(), (-1)])[1] > 90)
        # Now filter by Trues and get a good match
        try:
            matched_mps = matched_mps_in_wiki.where(lambda x: x==True).dropna().index
            # if there are multiple matches, match by exact surname
            for match in matched_mps:
                if ", Jr." in match:
                    if match.split("_")[0] == mp["full_name"]:
                        matched_mp = match
                        break
                elif match.split("_")[0].split(",")[0].split(" ")[-1] == mp["full_name"].split(",")[0].split(" ")[-1]:
                    matched_mp = match
                    break
            party = wikidata_mps["party"].loc[matched_mp]
            mp_id = wikidata_mps["mp_id"].loc[matched_mp]
        except UnboundLocalError:
            print(mp, matched_mps)
            raise UnboundLocalError
        return (party, mp_id, matched_mp)
    else:
        return (None, None, None)

if True:
    from multiprocessing import Pool

    # Create a pool of 8 processes
    pool = Pool(8)
    
    # Find 100% match, if it exists and add it to the MPs dataframe
    mps_wikidata = list(pool.map(match_mp, mps[["full_name", "Constituency"]].to_dict("records")))
    pool.close()
    pool.join()
    # Messy step to clean up the matches...
    mps_wikidata = list(zip(*[i if i != None else (None, None, None) for i in mps_wikidata]))
    mps = mps.assign(party_wikidata=mps_wikidata[0], mp_wikidata_id=mps_wikidata[1], mp_wikidata=mps_wikidata[2])

    # Map party names in wikidata to standardised party names
    party_abbr_wiki = {
    'Labour Party':"Labour",
     'Conservative Party':"Conservative",
     'Plaid Cymru':"Plaid Cymru",
     'Liberal Democrats':"Liberal Democrat",
     'Liberal Party':"Liberal Party",
     'Ulster Unionist Party':"UUP",
     'Scottish National Party':'Scottish National Party',
     'Democratic Unionist Party': "DUP",
     'Social Democratic Party': "SDP",
     'Social Democratic and Labour Party': "Social Democratic and Labour Party",
     'UK Independence Party': "UKIP",
     'Green Party': "Green",
     'Sinn Féin': "Sinn Féin",
     'Alliance Party of Northern Ireland': "Alliance",
     'Respect Party': "Respect",
     'Labour/Co-operative': "Labour/Co-operative",
     'Co-operative Party': "Labour/Co-operative"
    }
    # For MPs that do not have a party affiliation, use the party affiliation from mysociety
    mps.Party = mps.Party.fillna(mps["party_mysoc"])

# For the MPs that have wikidata info but no party, find their party
def scrape_mp_current_party(mp_id):
    from bs4 import BeautifulSoup
    """Look up MP's wikipedia page, then go down the table of data for MP and try to figure out current party.
    mp_id: the mp _ year of death key used in wikidata_mps data frame"""
    mp_wikidata_id = wikidata_mps.loc[mp_id]["mp_id"].split("/")[-1]
    parties = wikidata_mps.loc[mp_id]["party"]
    if len(parties) < 2:
        return parties
    wikipedia_name = requests.get("https://www.wikidata.org/w/api.php?action=wbgetentities&format=xml&props=sitelinks&ids={0}&sitefilter=enwiki".format(mp_wikidata_id))
    wikipedia_src = requests.get("https://en.wikipedia.org/wiki/" + str(BeautifulSoup(wikipedia_name.text, "html5lib").select("sitelink")[0]["title"]))
    print(wikipedia_src.url)
    for row in BeautifulSoup(wikipedia_src.text, "html5lib").select("table.infobox tr"):
        for party in parties:
            # Return first party that is found.
            # Assume that the party near the top of wikipedia data table is the MP's current party
            if (party in row.text) | (party.replace("Democrats", "Democrat") in row.text):
                return [party]
    return parties

if True:
    # Filter only the MPs that need wikidata info
    mps_filter = mps.party_wikidata.notnull() & mps.Party.isnull()
    mps.loc[mps_filter, "party_wikidata"] = mps.loc[mps_filter, "mp_wikidata"].apply(scrape_mp_current_party)

    # Assume that all Liberal MPs eventually became Liberal Democrats
    mps.loc[mps_filter, "party_wikidata"] = mps.loc[mps_filter, "party_wikidata"].apply(lambda x: ["Liberal Democrats"] if x == ["Liberal Democrats", "Liberal Party"] else x)

if False:
    # Flatten remaining party lists and for lists with several parties, just take the first one
    mps["party_wikidata"] = mps["party_wikidata"].apply(lambda x: None if x == None else party_abbr_wiki.get(x[0]))
    # If an MP doesn't have a party defined, then use one of the other sources to assign a party
    mps.Party = mps.Party.fillna(mps["party_wikidata"]).fillna(mps["party_women"])

    # Correct Sinn Féin spelling
    mps.loc[mps.Party == "Sinn Fein", "Party"] = "Sinn Féin"
    
    # Clean up some MP IDs that are different to the ones in speeches
    mps = mps.set_index("Person ID").reset_index()
    mps.loc[mps.query("mp_wikidata == 'Winnie Ewing_1929'").index[0], "Person ID"] = 22574
    mps.loc[mps.query("mp_wikidata == 'James Callaghan_1912'").index[0], "Person ID"] = 16877
    mps = mps.set_index("Person ID")
    
    # Save MPs to disk
    mps.to_hdf("list_of_mps.h5", "mps", mode="w")
else:
    mps = pd.read_hdf("list_of_mps.h5", "mps")

