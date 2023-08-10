import json
import pandas as pd

import numpy as np
import glob

members = pd.read_hdf("list_of_members.h5", "members")
members["full_name_"] = members["first_name"] + " " + members["last_name"]
members["last_name_lower"] = members["last_name"].str.lower()

state_names = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AS": "American Samoa",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District Of Columbia",
    "FM": "Federated States Of Micronesia",
    "FL": "Florida",
    "GA": "Georgia",
    "GU": "Guam",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MH": "Marshall Islands",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "MP": "Northern Mariana Islands",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PW": "Palau",
    "PA": "Pennsylvania",
    "PR": "Puerto Rico",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VI": "Virgin Islands",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming"
}

state_names = {v: k for k, v in state_names.items()}

def speech_to_df(speech_filename, chamber='house'):
    """
    This function reads in a json and converts it to a pandas dataframe.
    It also tries to find missing bioguide ids if they exist in the data.
    """
    from pandas.io.json import json_normalize
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process

    # Load json from file
    speech_json = json.load(open(speech_filename))
    # flatten json into dataframes
    speeches = json_normalize(speech_json["content"])
    metadata = json_normalize(speech_json).drop(["content"], axis=1)
    
    # Add filename to dataframe
    #speeches["file"] = speech_filename
    
   # Join dataframes together, duplicating info over all rows
    df = speeches.join(pd.DataFrame(np.repeat(metadata.values, len(speeches), axis=0), columns=metadata.columns))
    # Create a date column
    df["date"] = pd.to_datetime(df["header.day"] + " " + df["header.month"] + " " + df["header.year"])
    # Make lower case
    df["header.chamber"] = df["header.chamber"].str.lower()
    
    # Keep only one chamber and only only speech rows
    speech_df = df.loc[(df["kind"] == 'speech') & (df["header.chamber"] == chamber.lower())].copy()
    
    ## If there are any speeches without a bioguide id, let's try to find the appropriate person
    if ("speaker_bioguide" in speech_df.columns) and (len(speech_df.query("speaker_bioguide == 'None'")) > 0):
        # Assume that there is only one unknown speaker per dataframe
        unknown_speaker = speech_df.loc[speech_df["speaker_bioguide"] == "None"].iloc[0]
        # Get the title row which may contain a full name
        title_rows = df.query("kind=='title'")

        if len(title_rows) > 0:
            #print(speech_filename, " Num. rows: ", len(title_rows))
            title_row = title_rows.iloc[0]["text"].split("\n")[0].strip()

            # Get list of possible members based on last name
            if " of " in unknown_speaker["speaker"]:
                # Then we also have a location
                list_of_possible_members = members.loc[(members["term_start"] < unknown_speaker["date"]) &
                            (members["term_end"] > unknown_speaker["date"]) &
                            (members["state"] == state_names[unknown_speaker["speaker"].split(" of ")[1]]) &
                            (members["type"] == ("rep" if unknown_speaker["header.chamber"] == "House" else "sen")) &
                            (members["last_name_lower"] == (unknown_speaker["speaker"].split(" of ")[0].split(" ")[-1].lower()))]\
                            .set_index("full_name_")["bioguide_id"]
            else:
                list_of_possible_members = members.loc[(members["term_start"] < unknown_speaker["date"]) &
                            (members["term_end"] > unknown_speaker["date"]) &
                            (members["type"] == ("rep" if unknown_speaker["header.chamber"] == "House" else "sen")) &
                            (members["last_name_lower"] == (unknown_speaker["speaker"].split(" ")[-1].lower()))]\
                            .set_index("full_name_")["bioguide_id"]

            if len(list_of_possible_members) == 1:
                # Reverse match using name to get bioguide id and set that as the id for all missing rows
                speech_df.loc[speech_df["speaker_bioguide"] == "None", "speaker_bioguide"] = list_of_possible_members.iloc[0]
                speech_df["changed"] = True
            elif len(list_of_possible_members) > 1:
                best_match = process.extractOne(title_row, list_of_possible_members.index.tolist(), scorer=fuzz.token_set_ratio)
                if best_match[1] > 85:
                    # Good match so let's use it
                    # Reverse match using name to get bioguide id and set that as the id for all missing rows
                    speech_df.loc[speech_df["speaker_bioguide"] == "None", "speaker_bioguide"] = list_of_possible_members.to_dict()[best_match[0]]
                    speech_df["changed"] = True

    if len(speech_df) > 0:
        # Group speeches together if by the same person
        speech_df = speech_df.groupby(["speaker", "speaker_bioguide", "id", "doc_title", "date"])            .apply(lambda x: " ".join(x.text)).reset_index().rename(columns={0:"body"})

        return speech_df

get_ipython().run_cell_magic('time', '', 'from multiprocessing import Pool\n\nfiles = glob.glob("/media/Stuff/congressional-record/output/**/json/*", recursive=True)\n\nwith Pool(8) as pool:\n    speeches = pd.concat(list(pool.map(speech_to_df, files)), ignore_index=True).drop("index", axis=1)')

speeches.join(members[["bioguide_id", "first_name", "last_name", "gender"]]                                    .drop_duplicates().set_index("bioguide_id"), on="speaker_bioguide")

import bcolz

# Save data to hdf5
speeches.drop("body", axis=1).to_hdf("speeches_metadata.h5", "metadata", mode="w", format="table")

# Save speeches to bcolz array
bcolz.carray(speeches["body"].str.replace("\n", " "), rootdir="speeches.bcolz", chunklen=10000000, cparams=bcolz.cparams(cname="lz4hc"))

# Save files that have been processed so that in the future we can append to speeches dataframe without duplicating work
pd.Series(files).to_csv("processed_files.csv", index=False)

del speeches

