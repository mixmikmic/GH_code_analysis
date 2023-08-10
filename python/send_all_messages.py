from time import time, sleep
import json

USER_DATA_FILE = 'users_data.json'
MESSAGE_TIMES_FILE = 'msg_times.json'
SENT_MESSAGES_LOG_FILE = 'sent_msgs.log'
NUM_MESSAGES = 1000

def is_time_to_send_message(t1, t2, freq):
    secs = freq * 60
    if t1 - t2 > secs:
        return True
    return False

def load_user_data():
    with open(USER_DATA_FILE, 'r') as udfile:
        try:
            udata = udfile.read()
        except:
            print "Error in reading file : %s" % USER_DATA_FILE
            udata = {}
    return json.loads(udata)

def load_last_messages_info():
    try:
        with open(MESSAGE_TIMES_FILE, 'r') as mtfile:
            msgs_info = mtfile.read()
    except:
        msgs_info = {}
        with open(MESSAGE_TIMES_FILE, 'w') as mtfile:
            mtfile.write(json.dumps(msgs_info))
    return json.loads(msgs_info)

def save_last_message_times(msg_times):
    try:
        with open(MESSAGE_TIMES_FILE, 'w') as mtfile:
            mtfile.write(json.dumps(msg_times))
    except:
        print "Error in writing file : %s" % MESSAGE_TIMES_FILE

def log_sent_messages(messages_log):
    with open(SENT_MESSAGES_LOG_FILE, 'ab') as smfile:
        flwriter = csv.writer(
                smfile,
                delimiter=',',
                quoting=csv.QUOTE_MINIMAL,
                lineterminator='\n',
                )
        for msg in messages_log:
            flwriter.writerow(msg)

def get_last_msg_index_and_time(number, data):
    msg_time, msg_index = last_messages_info.get(number, (0, 0))
    
    if msg_time == 0:
        from random import randint
        msg_index = randint(0, NUM_MESSAGES)
    return msg_time, msg_index
            
def send_message(number, msg_index):
    print "Sending message %s to number : %s" % (msg_index, number)
    
def send_all_messages():
    """
    This function is called every minute
    to send messages to subscribed numbers
    """
    # dict of number -> time when last message was sent
    last_messages_info = load_last_messages_info()
    messages_log = []
    now = int(time())

    # read users file
    udata = load_user_data()
    for number, (status, freq) in ud.items():
        if status == 'subscriber':
            last_msg_time, last_msg_index =                     get_last_msg_index_and_time(number, last_messages_info)
            if is_time_to_send_message(now,
                                       last_msg_time,
                                       freq):
                
                send_message(number, last_msg_index + 1)
                last_message_times[number] = (now, last_msg_index + 1)
                messages_log.append([number, now, last_msg_index])
                
    save_last_message_times(last_message_times)
    log_sent_messages(messages_log)
    
def main():
    while True:
        send_all_messages()
        sleep(60)

if __name__ == "__main__":
    main()



fl = open('sdlfjs', 'a')
fl.write('yoohoo')
fl.close()

print open('sdlfjs', 'r').read()

import os
os.remove('sdlfjs')

from random import randint

print randint(0, 20)

import csv
get_ipython().magic('pinfo2 csv.writer')

import csv
with open('eggs.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    spamwriter.writerow([255, 'Spam', 'Lovely, Spam', 'Wonderful Spam'])

pp(open('eggs.csv', 'r').readlines())

import time
get_ipython().magic('pinfo2 time.sleep')



























































