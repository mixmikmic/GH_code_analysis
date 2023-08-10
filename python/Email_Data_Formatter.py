get_ipython().magic('pylab inline')
import mailbox
import pandas as pd

mb = mailbox.mbox('All mail Including Spam and Trash.mbox')

keys = ['Date', 'X-Gmail-Labels', 'X-GM-THRID']
message_list = []

for message in mb.itervalues():
    dmessage = dict(message.items())
    message_list.append({key:dmessage[key] if key in dmessage.keys() else '' for key in keys})

print len(message_list), 'messages'
print '**'*50
message_list[:3]

messages = pd.DataFrame(message_list)
messages.index = messages['Date'].apply(lambda x: pd.to_datetime(x, errors='coerce'))
messages.drop(['Date'], axis=1, inplace=True)
print messages.shape
messages.head()

conversation_list = []
threads = messages.groupby(by='X-GM-THRID')
print len(threads), 'threads total'

counts = threads.aggregate('count')['X-Gmail-Labels'].value_counts()
counts.plot(logy=True, linewidth=0, marker='.', alpha=.5)
plt.ylabel('Number of Threads')
plt.xlabel('Length of Thread')

for name, group in threads:
    if len(group) > 1:
        if 'Sent' in group['X-Gmail-Labels'].values:
            group.sort_index(inplace=True)
            tstart = group.index[0]
            tjoin = group[group['X-Gmail-Labels']=='Sent'].index[0]
            conversation_list.append({'tstart':tstart, 'tjoin':tjoin})

conversations = pd.DataFrame(conversation_list)
print conversations.shape
conversations.head()

delta = conversations['tjoin']-conversations['tstart']
days = 1.* delta.dt.total_seconds() / 3600 / 24
days.head()

days = days[days>0]
days = days.reset_index()[0]
days.head()

days.to_csv('days_to_join_conversation.csv')

