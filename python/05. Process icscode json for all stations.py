# Use SQLAlchemy to write each one to the database

from mylibrary.connections import Automapped_Base, conn, engine,session
from mylibrary.secrets import app_id, app_key
Stations = Automapped_Base.classes.all_stations

# Grab a list of all the ids of records which need to be matched
nlcs = session.query(Stations.nlc).filter(Stations.tfl_response != None).all()

import json
import pandas as pd
def process_icsCode(station):
   
    tfl_response_json = json.loads(station.tfl_response)
    
    return_object = {}
    
    if len(tfl_response_json["places"])==0:
        return_object["icscode"] = None
        return_object["icscode_status"] = "failed - no places in json"
        return return_object
    
    def contains_rail(x):
        return "national-rail" in x
    
    #Try to find a station of type "NaptanRailStation","NaptanRailAccessArea", sort by distance
    try:
        df = pd.DataFrame(tfl_response_json["places"]).sort("distance")
        df = df[df["stopType"].isin(["NaptanRailStation","NaptanRailAccessArea", "NaptanPublicBusCoachTram"])]
        df = df[df["placeType"] == "StopPoint"]
        df = df[df["modes"].apply(contains_rail)]
        df = df[pd.notnull(df["icsCode"])]
        return_object["icscode"] = df.iloc[0]["icsCode"]
        return_object["icscode_status"] = "ok"
        return return_object
    except:
        pass

    #Make a last ditch attempt if that didn't work - based on icscodes in a format that seem to represent valid stations.
    try:
        df = pd.DataFrame(tfl_response_json["places"]).sort("distance")
        f1 = df["icsCode"].astype(str).str[:2] == "10"
        f2 = df["icsCode"].astype(str).str[:2] == "90"
        df = df[f1|f2]

        return_object["icscode"] = df.iloc[0]["icsCode"]
        return_object["icscode_status"] = "ok"
        return return_object
    except:
        return_object["icscode"] = None
        return_object["icscode_status"] = "failed - during filtering"
        return return_object
    

    return_object["icscode"] = None
    return_object["icscode_status"] = "failed - after filtering, no match"
    return return_object

# Iterate through adding icscode
for nlc in nlcs:
    station = session.query(Stations).filter(Stations.nlc == nlc).one()
    ics_object = process_icsCode(station)
    station.icscode = ics_object["icscode"]
    station.icscode_status = ics_object["icscode_status"]
    session.add(station)
    session.commit()



