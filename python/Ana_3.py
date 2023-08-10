import csv
# import CSV file which is generated in last analysis, and turn it into matrix
email_matrix = []
with open('ana_2/email_count.csv', 'r') as f:
    reader = csv.reader(f)
    email_matrix = [[x for x in row] for row in reader]
print(len(email_matrix))
print(len(email_matrix[0]))

# for each row in matrix, find the indices of maximum number in that row
# index of most_emails refers to id of sender
# value of most_emails refers to id of receiver
most_emails= []
for num_list in email_matrix:
    max_value = max(num_list)
    max_index = num_list.index(max_value)
    most_emails.append(max_index)
print(len(most_emails))

import csv
# from this CSV we get id and relating email address
id_dict = {}
with open('ana_2/email_id.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id_dict[int(row['ID'])] = row['NAME']
print(len(id_dict))

output_path = 'ana_3/most_email_friend.csv'
with open(output_path, 'w') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames = ["FROM", "TO", "EMAIL_NUM"])
        writer.writeheader()
        row = 0
        for x in most_emails:
            em_num = int(email_matrix[row][x])
            # some users send no emails to people in this data set
            if em_num == 0:
                continue
            # row refers to id of sender, I only want the name part, not the whole email address
            from_user = id_dict[row].split('@')[0]
            # x refers to id of receiver
            to_user = id_dict[x].split('@')[0]
            row += 1
            writer.writerow({'FROM': from_user,'TO': to_user, 'EMAIL_NUM': str(em_num)})



