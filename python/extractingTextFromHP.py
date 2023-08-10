from unidecode import unidecode
dataComplete = []
for i in range(1,8):
    with open('dataset/book_' + str(i) + '.txt', 'r') as myfile:
        data=myfile.read()
        data = data.decode('utf-8')
        data = unidecode(data)
        dataComplete = dataComplete + data.split()

len(dataComplete)

import string
from unidecode import unidecode
with open('dataset/book_1.txt', 'r') as myfile:
    data=myfile.read()
    data = data.decode('utf-8')
    data = unidecode(data)
    data = data.translate(None, string.punctuation)

data.lower().split()



