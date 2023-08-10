import glob
import os
import csv
from pathlib import Path
from collections import Counter

p = Path(os.getcwd())
email_path = str(p.parent) + '/data/enron/maildir/*/sent/*.'

# first, store the users that is in data set
files = glob.glob(email_path)
sent_email_address_list = []
for file in files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i == 2:
                    sent_email_address_list.append(line[6:-1])
                    break
print(len(set(sent_email_address_list)))

# transfer this user set into a dictionary that relate an email address to an ID
email_dict = {}
index = 0
for em in set(sent_email_address_list):
    email_dict[em] = index
    index += 1
print(len(email_dict))

# we have 121 users in this data set
# thus a 121*121 matrix will be created, each of whose element represents number of emails between two users
# row indices are IDs of senders
# column indices are IDs of receivers
email_matrix = [[0 for x in range(len(email_dict))] for y in range(len(email_dict))] 

# if a email is sent from A to B, this email will be found in A's sent-box and in B's in-box
# thus read only from 'send' folders to prevent duplicates
from_email = ''
# there may be multiple receivers, so create a list to store them
to_email_list = []
for file in files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                # this line refers to "From: xxx"
                if i == 2:
                    from_email = line[6:-1]
                # starting from this line("To: xxx"), there may be multiple lines of reveivers
                # after "To: xxx", the line "Subject: xxx" indicates the end of receivers
                elif i > 2 and not line.startswith('Subject'):
                    # extract only email addresses from receivers, then add them to receiver list
                    to_email_list.extend(line.replace('To: ', '').strip().split(', '))
                elif line.startswith('Subject'):
                    # all reveivers stored in list, update the matrix
                    row = email_dict[from_email]
                    for em in to_email_list:
                        if em == '' or em not in email_dict:
                            continue
                        else:
                            col = email_dict[em]
                            email_matrix[row][col] += 1
                    # clear the list for next email
                    to_email_list = []
                    from_email = ''
                    break
print(len(email_matrix))

# generate a CSV showing email addresses and relating IDs
output_path = 'ana_2/email_id.csv'
with open(output_path, 'w') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames = ["NAME","ID"])
        writer.writeheader()
        for em,em_id in email_dict.items():
            writer.writerow({'NAME': em,'ID': str(em_id)})

# generate a CSV of the matrix
with open("ana_2/email_count.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(email_matrix)



