import os
from IPython.display import clear_output

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
define pastas de trabalho
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
path = 'C:/Users/marcelo.ribeiro/Documents/textfiles-corrected-renamed2/' #path_output

onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and not f.startswith('.')]
onlyfiles.sort()

onlyfiles[:4]

corpus_counter = []
percentil = int(len(onlyfiles)/100)
for txt in onlyfiles:
    with open(os.path.join(path,txt), 'r', encoding='utf-8') as f:
        count = onlyfiles.index(txt)
        if count % (percentil-1) == 0: clear_output()
        if count % percentil == 0: print(int(count/percentil),'% done')
        text = f.read()
        text = text.lower()
        text = text.split()
        text = [t.strip(string.punctuation) for t in text]
        for word in text: 
            if len(word) > 0: corpus_counter.append(word)

len(corpus_counter)



