# pip3 install slacker
from slacker import Slacker

token = 'xoxo-your-token'
slack = Slacker(token)
slack.chat.post_message(channel='채널명 혹은 채널 id', text='message test', as_user=False)

# Channel 확인 : id값 확인 가능
slack.channels.list().body

# Group 확인 : id값 확인 가능
slack.groups.list().body



