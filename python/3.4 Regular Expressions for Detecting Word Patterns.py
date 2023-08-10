import nltk
import re
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
wordlist

[w for w in wordlist if re.search('ed$', w)]

[w for w in wordlist if re.search('^..j..t..$', w)]

[w for w in wordlist if re.search('..j..t..', w)]

sum(1 for w in wordlist if re.search('^e-?mail$', w))

[w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)]

chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
[w for w in chat_words if re.search('^m+i+n+e+$', w)]

[w for w in chat_words if re.search('^[ha]+$', w)]

[w for w in chat_words if re.search('^m*i*n+e*$', w)]

[w for w in chat_words if re.search('^[^aeiouAEIOU]+$', w)]

wsj = sorted(set(nltk.corpus.treebank.words()))
[w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)]

[w for w in wsj if re.search('^[A-Z]+\$$', w)]

[w for w in wsj if re.search('^[0-9]{4}$', w)]

[w for w in wsj if re.search('^[0-9]+-[a-z]{3,5}$', w)]

[w for w in wsj if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$', w)]

[w for w in wsj if re.search('(ed|ing)$', w)]

[w for w in wsj if re.search('^w(i|e|ai|oo)t$', w)]



