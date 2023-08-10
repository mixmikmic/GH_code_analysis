from bs4 import BeautifulSoup,SoupStrainer
import re

doc = '''Bob reports <a href="http://www.bob.com/">success</a>
with his plasma breeding <a
href="http://www.bob.com/plasma">experiments</a>. <i>Don't get any on
us, Bob!</i>

<br><br>Ever hear of annular fusion? The folks at <a
href="http://www.boogabooga.net/">BoogaBooga</a> sure seem obsessed
with it. Secret project, or <b>WEB MADNESS?</b> You decide!'''

links = SoupStrainer('a')
[tag for tag in BeautifulSoup(doc,"lxml",parse_only=links)]

linksToBob = SoupStrainer('a', href=re.compile('bob.com/'))
[tag for tag in BeautifulSoup(doc,"lxml", parse_only=linksToBob)]

mentionsOfBob = SoupStrainer(text=re.compile("Bob"))
[text for text in BeautifulSoup(doc,"lxml", parse_only=mentionsOfBob)]

allCaps = SoupStrainer(text=lambda t:t.upper()==t)
[text for text in BeautifulSoup(doc,"lxml", parse_only=allCaps)]



