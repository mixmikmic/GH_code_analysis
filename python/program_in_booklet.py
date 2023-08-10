get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 99999;\n//increase max size of output area')

import json
import datetime as dt
from   operator import itemgetter

from IPython.display import display, HTML
from IPython.nbconvert.filters.markdown import markdown2html

talk_sessions = json.load(open('talk_abstracts.json'))
talks_admin_url = 'https://ep2015.europython.eu/admin/conference/talk'

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
    abstract  = talk['abstracts'][0]
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
    show('<p>{}</p>'.format(markdown2html(abstract)))
    show('<br/>')

session_names = ['Keynotes', 'Talks', 'Trainings', 'Help desks',
                 'EuroPython sessions', 'Other sessions', 'Poster sessions']

for session_name in session_names:
    show('<h1>{}</h1>'.format(session_name))
    
    talks = talk_sessions[session_name]
    talks = [talks[talk_id] for talk_id in talks]
    talks = sorted(talks, key=itemgetter('title'))
    for talk in talks:
        show_talk(talk, show_duration=False, show_link_to_admin=False)



