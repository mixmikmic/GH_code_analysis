inputFileName = r'cats_suggestions.csv'
outputFileName = r'cats_suggestions.sql.txt'
insertString = 'INSERT INTO cats.suggestions (like_id, user_id, video_id, "time")'

with open(inputFileName, 'r') as fo:
    with open(outputFileName, 'w') as fw:
        i = 0
        for line in fo:
            if (i > 0): # Skip header line
                line = line.rstrip('\n')
                v = str.split(line, ',')
                fw.write('{} VALUES ('.format(insertString))
                for vi in v[:-1]:
                    if vi.isnumeric(): # Check if it is a number
                        fw.write('{},'.format(vi))
                    else:
                        fw.write('\'{}\','.format(vi))
                        
                if vi.isnumeric(): # Check if it is a number
                    fw.write('{}'.format(v[-1]))
                else:
                    fw.write('\'{}\''.format(v[-1]))
                    
                fw.write(');\n')
            i += 1
fw.close()
fo.close()



