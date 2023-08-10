from mylibrary.connections import Automapped_Base, session
Stations = Automapped_Base.classes.all_stations

# ids = session.query(Stations.nlc).filter(Stations.london_or_gb == "gb").all()
nlcs = session.query(Stations.nlc).all()

from mylibrary.secrets import app_id, app_key

# Dump all the json to the database first.  Then process it later
# Iterate through adding icscode
import requests

from mylibrary.tfl_helpers import status_of_tfl_response_places

def get_icsCode(lat,lng):
    my_dict = {"lat": lat,
        "lng": lng,
        "id": app_id,
        "key": app_key}
    
    full_str = "".join([r"https://api.tfl.gov.uk/Place?",
    r"lat={lat}",
    r"&lon={lng}",
    r"&radius=1000",
    r"&includeChildren=False",
    r"&app_id={id}",
    r"&app_key={key}"])
    
    url = full_str.format(**my_dict)
    r = requests.get(url)
    
    message = status_of_tfl_response_places(r.content)
    
    return_object = {"json": r.content, "request_url": url, "tfl_message": message}
    return return_object
    

for nlc in nlcs:
    station = session.query(Stations).filter(Stations.nlc == nlc).one()
    ics_object = get_icsCode(station.lat, station.lng)
    station.tfl_request = ics_object["request_url"]
    station.tfl_response = ics_object["json"]
    station.tfl_message = ics_object["tfl_message"]
    session.add(station)
    session.commit()

