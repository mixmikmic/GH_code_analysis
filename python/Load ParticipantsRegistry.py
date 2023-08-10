import io
import json
import os.path as op
from itertools import chain
from collections import defaultdict

from eptools.people import (ParticipantsRegistry,
                            fetch_ticket_profiles,
                            contact_regex2,
                            parse_contact,
                            )

from eptools.talks import (fetch_talks_json, 
                           get_speaker_type,
                           get_talk_code,
                           get_type_speakers,
                          )

# declare the parameters
fetch_data = False
conf = 'ep2016'
host = 'europython.io'

talks_status   = 'proposed' # in the final this should be 'accepted'
talks_json     = 'talks_with_votes.json'
profiles_json  = 'profiles.json'
organizers_txt = 'organizers.txt'
volunteers_txt = 'volunteers.txt'
epsmembers_txt = 'epsmembers.txt'

def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)


def load_id_json(json_path):
    return [item for eid, item in json.load(open(json_path)).items()]


def read_lines(txt_file):
    with io.open(txt_file, 'rt', encoding='utf-8') as f:
        lines = f.readlines()
    return [l.strip() for l in lines]


def read_names(txt_file):
    lines = read_lines(txt_file)
    return [(name.split(' ')[0], ' '.join(name.split(' ')[1:])) for name in lines]


def read_contacts(txt_file=organizers_txt):
    return [parse_contact(line, regex=contact_regex2) for line in read_lines(txt_file)]

# fetch the data
if fetch_data:
    _ = fetch_ticket_profiles(profiles_json, conf=conf)
    _ = fetch_talks_json     (talks_json,    conf=conf, status=talks_status, host=host, with_votes=True)

# load the data
talks = {}
people = load_id_json(profiles_json)
type_talks = dict(json.load(open(talks_json)).items())
_ = [talks.update(talkset) for ttype, talkset in type_talks.items()]

# speakers and trainers
type_speakers = get_type_speakers(talks)

organizers = read_contacts(organizers_txt)
volunteers = read_contacts(volunteers_txt)
epsmembers = read_contacts(epsmembers_txt)

# build the cake
pr = ParticipantsRegistry(people)

for stype, emails in type_speakers.items():
    pr.set_people_role(emails, stype)

pr.set_people_role([p[2] for p in organizers], 'organizer')
pr.set_people_role([p[2] for p in volunteers], 'volunteer')
pr.set_people_role([p[2] for p in epsmembers], 'epsmember')

list(pr.get_roles_of('alexsavio@gmail.com'))

