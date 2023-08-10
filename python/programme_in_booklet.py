get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 99999;\n//increase max size of output area')

import json
import datetime as dt
from   operator import itemgetter
from   collections import OrderedDict
from   datetime import datetime

from IPython.display import display, HTML
from nbconvert.filters.markdown import markdown2html

talk_sessions = json.load(open('accepted_talks.json'))
talks_admin_url = 'https://ep2016.europython.eu/admin/conference/talk'

sessions_talks = OrderedDict()

# remove the IDs from the talks
for name, sess in talk_sessions.items():
    sessions_talks[name] = [talk for tid, talk in sess.items()]


# add 'start' time for each talk
for name, talks in sessions_talks.items():
    for talk in talks:
        tr = talk['timerange']
        if not tr:
            talk['start'] = datetime.now()
        else:
            talk['start'] = datetime.strptime(tr.split(',')[0].strip(), "%Y-%m-%d %H:%M:%S")


from operator import itemgetter
for sess, talks in sessions_talks.items():
    sessions_talks[sess] = sorted(talks, key=itemgetter('start', 'track_title'))

show = lambda s: display(HTML(s))

def ordinal(n):
    if 10 <= n % 100 < 20:
        return str(n) + 'th'
    else:
        return  str(n) + {1 : 'st', 2 : 'nd', 3 : 'rd'}.get(n % 10, "th")

def talk_schedule(start, end):

    input_format  = "%Y-%m-%d %H:%M:%S"
    output_format_day = "%A, %B"
    output_format_time = "%H:%M"
    
    output_date = lambda d: "{} {} at {}".format(d.strftime(output_format_day), 
                                                 ordinal(int(d.strftime('%d'))),
                                                 d.strftime(output_format_time))
    
    start_date = dt.datetime.strptime(start, input_format)
    end_date   = dt.datetime.strptime(end  , input_format)

    return output_date(start_date), output_date(end_date)


def show_talk(talk, show_duration=True, show_link_to_admin=True):
    
    speakers  = talk['speakers']
    title     = talk['title']
    abstract  = talk['abstract_long'][0]
    room      = talk.get('track_title', '').split(', ')[0]
    timerange = talk.get('timerange', '').split(';')[0]
    
    show('<h2>{}</h2>'.format(title))
    
    if show_link_to_admin:
        talk_admin_url = talks_admin_url + '/{}'.format(talk['id'])
        show('<a href={0}>{0}</a>'.format(talk_admin_url))
    
    if show_duration:
        duration = '{} mins.'.format(talk['duration'])
    else:
        duration = ''

    timerange = talk['timerange'].split(';')[0]
    try:
        start, end = talk_schedule(*timerange.split(', '))
    except:
        start, end = ('', '')

    if start:
        schedule  = '<p>'
        schedule += '{} in {}'.format(start, room)
        if show_duration:
            schedule += ' ({})'.format(duration)
        schedule += '</p>'

        show(schedule)
    
    show('<h3><i>{}</i></h2>'.format(speakers))
    #show('<p>{}</p>'.format(markdown2html(abstract)))
    show('<br/>')

#session_names = ['Keynotes', 'Talks', 'Trainings', 'Help desks',
#                 'EuroPython sessions', 'Other sessions', 'Poster sessions']

from collections import OrderedDict

# session_names = [('Keynote',            'Keynotes'),
#                  ('talk',               'Talks'),
#                  ('training',           'Tutorials'), 
#                  ('poster',             'Posters'),
#                  ('helpdesk',           'Helpdesks'),
#                  ('EPS session',        'EuroPython Society Sessions'),
#                  ('Recruiting session', 'Recruiting'),
#                  ('interactive',        'Interactive sessions'),
#                  ('Reserved slot',      'TALKS'),
#                  ('Lightning talk',     'TALKS'),
#                  ('Closing session',    'Closing session'),                 
#                 ]

session_names = [('Keynote',            'Keynotes'),
                ]

session_names = OrderedDict(session_names)

for name, title in session_names.items():
    show('<h1>{}</h1>'.format(title))
    
    talks = sessions_talks[name]
    for talk in talks:
        show_talk(talk, show_duration=False, show_link_to_admin=False)



