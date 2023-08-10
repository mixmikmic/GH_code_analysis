import json

from eptools.people import fetch_users

fetch_data = False
talks_json = 'talks_with_votes.json'
users_json = 'users.json'
talks      = {}

if fetch_data:
    _ = fetch_users     (users_json)
    _ = fetch_talks_json(talks_json, conf=conf, status=talks_status, host=host, with_votes=True)

users      = dict(json.load(open(users_json)).items())
type_talks = dict(json.load(open(talks_json)).items())
_ = [talks.update(talkset) for ttype, talkset in type_talks.items()]

low_vote   = 3.0
low_voters = set()

for tid, talk in talks.items():
    talk_low_voters = [uid for pair in talk['user_votes'] for uid, v in pair.items() if v < low_vote]
    low_voters |= set(talk_low_voters)

_ = [print(users[uid]['email']) for uid in low_voters]

