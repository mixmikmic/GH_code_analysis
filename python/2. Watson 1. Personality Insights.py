# The code was removed by DSX for sharing.

text="""
DIE A HAPPY MAN SONG

Baby last night was hands down
One of the best nights
That I've had no doubt
Between the bottle of wine
And the look in your eyes and the Marvin Gaye
Then we danced in the dark under September stars in the pourin' rain.

And I know that I can't ever tell you enough
That all I need in this life is your crazy love
If I never get to see the Northern Lights
Or if I never get to see the Eiffel Tower at night
Oh if all I got is your hand in my hand
Baby I could die a happy man
Happy man, baby, hmmmm.


Baby and that red dress brings me to my knees
Oh but that black dress makes it hard to breathe
You're a saint, you're a goddess, the cutest, the hottest, the masterpiece
It's too good to be true, nothing better than you
In my wildest dreams.

And I know that I can't ever tell you enough
That all need in this life is your crazy love
If I never get to see the Northern Lights
Or if I never get to see the Eiffel Tower at night
Oh if all I got is your hand in my hand
Baby I could die a happy man, yeah.

I don't need no vacation, no fancy destination
Baby you're my great escape
We could stay at home, listen to the radio
Or dance around the fireplace.

And if I never get to build my mansion in Georgia
Or drive a sports car up the coast of California
Well if all I got is your hand in my hand
Baby I could die a happy man
Baby I could die a happy man
Oh I could die a happy man
You know I could, girl
I could die, I could die a happy man, hmmmm.
"""

import requests
import json

url = credentials_1['url']+"/v3/profile?version=2016-10-20&consumption_preferences=true&raw_scores=true"

headers = {'Content-Type': 'text/plain'}
auth=(credentials_1['username'], credentials_1['password'])
data = {'text':text}

response=requests.post(url,headers=headers,auth=auth, data=data )
parsed = json.loads(response.text)
print json.dumps(parsed, indent=4, sort_keys=True)


get_ipython().system('pip install watson-developer-cloud')

import json
from os.path import join, dirname
from watson_developer_cloud import PersonalityInsightsV3

"""
The example returns a JSON response whose content is the same as that in
   ../resources/personality-v3-expect2.txt
"""

personality_insights = PersonalityInsightsV3(
    version='2016-10-20',
    username='YOUR SERVICE USERNAME',
    password='YOUR SERVICE PASSWORD')

with open(join(dirname(__file__), '../resources/personality-v3.json')) as         profile_json:
    profile = personality_insights.profile(
        profile_json.read(), content_type='application/json',
        raw_scores=True, consumption_preferences=True)

    print(json.dumps(profile, indent=2))

