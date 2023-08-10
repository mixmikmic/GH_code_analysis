import csv

with open ('../data/mlb.csv', 'r') as f:
    
    # create a reader object
    reader = csv.DictReader(f)
    
    # print the fieldnames
    print(reader.fieldnames)
    
    # loop over the reader object and print the 
    for row in reader:
        print(row['TEAM'], row['NAME'], row['SALARY'])

new_mlb_players = [
    {'NAME': 'Jeff Tweedy', 'POS': 'SP', 'SALARY': 1000000, 'START_YEAR': 2015, 'END_YEAR': 2019, 'YEARS': 5},
    {'NAME': 'John Stirratt', 'POS': '1B', 'SALARY': 950000, 'START_YEAR': 2015, 'END_YEAR': 2019, 'YEARS': 5},
    {'NAME': 'Nels Cline', 'POS': 'C', 'SALARY': 900000, 'START_YEAR': 2015, 'END_YEAR': 2019, 'YEARS': 5},
    {'NAME': 'Pat Sansone', 'POS': 'SS', 'SALARY': 850000, 'START_YEAR': 2015, 'END_YEAR': 2019, 'YEARS': 5},
    {'NAME': 'Mikael Jorgensen', 'POS': 'RP', 'SALARY': 800000, 'START_YEAR': 2015, 'END_YEAR': 2019, 'YEARS': 5},
    {'NAME': 'Glenn Kotche', 'POS': '3B', 'SALARY': 750000, 'START_YEAR': 2015, 'END_YEAR': 2019, 'YEARS': 5},
]

with open('new-mlb-players.csv', 'w', newline='') as f:
    headers = ['NAME', 'POS', 'SALARY', 'START_YEAR', 'END_YEAR', 'YEARS']
    writer = csv.DictWriter(f, fieldnames=headers)
    
    writer.writeheader()
    writer.writerows(new_mlb_players)

with open('../data/mlb.csv', 'r') as infile, open('royals.csv', 'w') as outfile:
    reader = csv.DictReader(infile)
    headers = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=headers)
    
    writer.writeheader()
    
    for row in reader:
        if row['TEAM'] == 'KC':
            writer.writerow(row)

