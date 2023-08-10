import csv
import re

orig_file_name = 'rahm_spending.csv'
csv_file = open(orig_file_name, 'rb')
infile = csv.reader(csv_file)

headers = infile.next()

def cleaner(row):

    lastonlyname = row[0].upper()

    amount = float(row[3].replace('$', '').strip())

    if row[6] in ['CHGO', 'CHCAGO']:
        city = 'CHICAGO'
    else:
        city = row[6].replace('&NBSP;', ' ')
        
    if len(row[8]) == 4:
        zip = '0{}'.format(row[8])
    else:
        zip = row[8]

    p_split = re.split('-|/', row[10])
    if len(p_split) > 1:
        main_purpose = p_split[0].strip()
        purpose_extra = p_split[1].strip()
    else:
        main_purpose = row[10]
        purpose_extra = ''
    problem_words = ['FEE', 'FEES', 'COST', 'COSTS', 'EXPENSE']
    purpose_words = main_purpose.split()
    for word in purpose_words:
        if word in problem_words:
            loc = purpose_words.index(word)
            purpose_words.pop(loc)
            purpose_words.insert(loc, 'EXPENSES')
    main_purpose = ' '.join(purpose_words)

    cleaned_row = [lastonlyname, row[1], row[2], amount, row[4], row[5], city, row[7], zip, row[9], main_purpose, purpose_extra, row[11]]
    return cleaned_row

headers.insert(headers.index('PURPOSE') + 1, 'DETAIL')

clean_file_name = 'rahm_spending_clean.csv'
with open(clean_file_name, 'wb') as outfile:
    output = csv.writer(outfile)
    output.writerow(headers)
    for row in infile:
        # Here's where we can weed out non-expenditures from hitting our clean file.
        if row[9] == 'EXPENDITURE':
            output.writerow(cleaner(row))

print('All done!')
outfile.close()

