import tvm_users_scatter as tus
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import numpy as np

q_current = '''
select
    ch.category,
    ch.name as channel,
    sum(ss.clip_duration)/60 as tvm,
    count(distinct anon_user_id) users
from test_pluto_clip_start_stop_times ss
join pluto_channels ch
    on ch.id = ss.channel_id
    and ch.direct_only = true
    and ss.request_date >= current_date - 30
    and ss.platform not in ('Web','Unknown','Embed','Desktop')
group by 1,2
'''

q_previous = '''
select
    ch.category,
    ch.name as channel,
    sum(ss.clip_duration)/60 as tvm,
    count(distinct anon_user_id) users
from test_pluto_clip_start_stop_times ss
join pluto_channels ch
    on ch.id = ss.channel_id
    and ch.direct_only = true
    and ss.request_date between current_date - 60 and current_date - 31
    and ss.platform not in ('Web','Unknown','Embed','Desktop')
group by 1,2
'''

df = tus.queryData(query=q_current)
df = tus.cleanDF(df)

reload(tus)
p = tus.createViz(df)

# df2 = tus.queryData(query=q_previous)
# df2 = tus.cleanDF(df2)

# reload(tus)
# p = tus.createViz(df)

