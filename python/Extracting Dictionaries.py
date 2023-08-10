import csv

ls

csvFile = "inquirerbasic.csv"

listOfRows = []
with open(csvFile, 'r') as file: # This makes sure that file is closed after reading
    data = csv.reader(file)
    for row in data:
        listOfRows.append(row)
file.closed

categories = listOfRows[0]
for i in range(2,len(categories)): # Note that we start at 2 as the first to labels are not categories
    print(str(i), ": ", categories[i])

category = 2

words = []
for row in listOfRows[1:]: # We iterate over the rows skipping the header row
    if row[category] != "":
        words.append(row[0])

print(str(len(words)) + " words in category:" + " " + categories[category])
words[:50]

cleanedWords = []
theLastWord = ""
for word in words:
    if "#" in word:
        if theLastWord not in word:
            cleanedWords.append(word[:-2])
            theLastWord = (word[:-2])
    else:
        cleanedWords.append(word)
        theLastWord = word

print(str(len(cleanedWords)) + " words in cleaned category:" + " " + categories[category])
cleanedWords[:50]

nameOfDict = categories[category] + ".dictionary.txt"

with open(nameOfDict, "w") as fileToWrite:
    for word in cleanedWords:
        fileToWrite.write(word.lower() + "\n")
    
print("Done")



