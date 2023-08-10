import pandas as pd
import requests
from bs4 import BeautifulSoup, SoupStrainer
import time
import random
import re

url = 'http://www.successories.com/iquote/author/291/abraham-lincoln-quotes/' + 'i'

r = requests.get(url)

content = r.text

r

soup = BeautifulSoup(content, 'html.parser')

print soup.find(name='div', attrs={'class':'quote'}).text

quote_list = []

for quote in soup.find_all(name='div', attrs={'class':'quote'}):
    quote_list.append(quote.text)

quote_list

for i in range(1, 45):
    url = 'http://www.successories.com/iquote/author/291/abraham-lincoln-quotes/' + 'i'

quote_url = []
for i in range(1,119):
    quote_url.append('http://www.successories.com/iquote/author/11/mark-twain-quotes/' + str(i))

quote_url[-1]

twain_quotes = pd.DataFrame(columns=['quote'])

for url in quote_url:
    r = requests.get(url)
    html = r.text
    soup = BeautifulSoup(html, 'html.parser')
    for quote in soup.find_all(name='div', attrs={'class':'quote'}):
        q = quote.text
        twain_quotes.loc[len(twain_quotes)]=[q]
    n = random.randint(0,15)
    time.sleep(n)

twain_quotes.shape

twain_quotes.to_csv('twain_quotes.csv', encoding='utf-8')

lincoln_quotes = pd.DataFrame(columns=['quote'])

quote_url2 = []
for i in range(1,45):
    quote_url2.append('http://www.successories.com/iquote/author/291/abraham-lincoln-quotes/' + str(i))

for url in quote_url2:
    r = requests.get(url)
    html = r.text
    soup = BeautifulSoup(html, 'html.parser')
    for quote in soup.find_all(name='div', attrs={'class':'quote'}):
        q = quote.text
        lincoln_quotes.loc[len(lincoln_quotes)]=[q]
    n = random.randint(0,15)
    time.sleep(n)

lincoln_quotes.to_csv('lincoln_quotes.csv', sep='|', encoding='utf-8')

quote_url3 = []
for i in range(1, 91):
    quote_url3.append('http://www.successories.com/iquote/author/192/oscar-wilde-quotes/'+str(i))

wilde_quotes = pd.DataFrame(columns=['quote'])

for url in quote_url3:
    r = requests.get(url)
    html = r.text
    soup = BeautifulSoup(html, 'html.parser')
    for quote in soup.find_all(name='div', attrs={'class':'quote'}):
        q = quote.text
        wilde_quotes.loc[len(wilde_quotes)]=[q]
    n = random.randint(0,15)
    time.sleep(n)

wilde_quotes.to_csv('wilde_quotes.csv', sep='|', encoding='utf-8')

url = 'https://www.mcsweeneys.net/articles/it-takes-a-village-of-9-and-11-year-old-girls-to-raise-my-child'

r = requests.get(url)

html = r.text

soup = BeautifulSoup(html, 'html.parser')

for item in soup.find_all("div", attrs={"class":"articleBody"}):
    print item.text

urls5 = [
    'https://www.mcsweeneys.net/articles/it-takes-a-village-of-9-and-11-year-old-girls-to-raise-my-child',
'https://www.mcsweeneys.net/articles/im-one-of-those-nice-guys',
'https://www.mcsweeneys.net/articles/thanks-cindy-for-making-eye-contact-through-the-bathroom-stall-and-making-it-super-awkward-during-the-department-productivity-meeting',
'https://www.mcsweeneys.net/articles/im-the-person-everyone-is-trying-to-convince-not-to-vote-for-donald-trump-on-social-media',
'https://www.mcsweeneys.net/articles/16-useful-mnemonics-for-commonly-misspelled-words-from-a-desperate-writer-who-told-us-he-knew-20',
'https://www.mcsweeneys.net/articles/are-you-the-next-rock-star-social-media-manager-whos-willing-to-literally-die-for-content',
'https://www.mcsweeneys.net/articles/the-prose-edda-for-bostonians-gylfaginning-part-xxi',
'https://www.mcsweeneys.net/articles/im-looking-for-a-candidate-who-tells-it-like-it-is',
'https://www.mcsweeneys.net/articles/the-fourteenth-batch-2016',
'https://www.mcsweeneys.net/articles/your-play-has-been-accepted-into-the-theater-and-cheese-festival-of-yakima-wa',
'https://www.mcsweeneys.net/articles/shrimp-whistle',
'https://www.mcsweeneys.net/articles/i-want-to-apply-for-the-position-of-rent-seeking-capitalist',
'https://www.mcsweeneys.net/articles/can-you-believe-donald-trump-did-that-thing',
'https://www.mcsweeneys.net/articles/trump-campaign-memo-regarding-subway-sandwiches',
'https://www.mcsweeneys.net/articles/this-is-the-year-the-cubs-win-the-world-series-and-thus-the-prophecy-shall-be-fulfilled',
'https://www.mcsweeneys.net/articles/i-am-a-writer',
'https://www.mcsweeneys.net/articles/how-to-talk-to-a-woman-with-no-headphone-jack',
'https://www.mcsweeneys.net/articles/a-college-applicants-love-letter',
'https://www.mcsweeneys.net/articles/welcome-to-hoodwink-the-advertising-agency-that-is-not-an-advertising-agency',
'https://www.mcsweeneys.net/articles/congratulations-your-essay-has-been-accepted-by-loan-repayment-success-story-quarterly',
'https://www.mcsweeneys.net/articles/the-foreshortened-career-arc-of-a-contingent-writing-instructor-by-the-numbers',
'https://www.mcsweeneys.net/articles/tentative-outline-for-the-remaining-2016-election-news-coverage',
'https://www.mcsweeneys.net/articles/a-chapter-from-really-hot-love-a-harlequin-romance-written-by-13-year-old-thomas-freeman',
'https://www.mcsweeneys.net/articles/wedding-vows-to-my-work-wife',
'https://www.mcsweeneys.net/articles/were-sorry-we-failed-to-deliver-nude-photos-of-arthur-miller',
'https://www.mcsweeneys.net/articles/9-11',
'https://www.mcsweeneys.net/articles/the-art-of-asking-a-question-to-a-literary-festival-panel',
'https://www.mcsweeneys.net/articles/the-prose-edda-for-bostonians-gylfaginning-part-xx',
'https://www.mcsweeneys.net/articles/excerpts-from-the-gilmore-girls-revival-script-which-seem-to-indicate-that-aliens-play-a-large-role-in-the-show',
'https://www.mcsweeneys.net/articles/shira-and-her-boss-take-care-of-business',
'https://www.mcsweeneys.net/articles/its-decorative-gourd-season-motherfuckers',
'https://www.mcsweeneys.net/articles/spring-forward-fall-into-perpetual-darkness',
'https://www.mcsweeneys.net/articles/our-tiny-home-is-revolutionizing-how-my-wife-and-i-fight',
'https://www.mcsweeneys.net/articles/im-mad-as-hell-and-im-only-going-to-put-up-with-it-for-another-ten-or-fifteen-years',
'https://www.mcsweeneys.net/articles/is-he-hot-or-is-he-just-holding-their-eyes-were-watching-god',
'https://www.mcsweeneys.net/articles/miners-lamp-tag-and-cycling-end-caps',
'https://www.mcsweeneys.net/articles/the-tracklist-for-my-all-white-entirely-male-punk-bands-forthcoming-trump-era-opus',
'https://www.mcsweeneys.net/articles/an-honest-intern-application-cover-letter',
'https://www.mcsweeneys.net/articles/david-your-latest-marketing-report-brought-me-to-tears',
'https://www.mcsweeneys.net/articles/on-being-way-too-black',
'https://www.mcsweeneys.net/articles/i-went-to-a-trump-rally-what-i-found-there-was-a-bunch-of-other-journalists-already-writing-this-article',
'https://www.mcsweeneys.net/articles/from-the-therapy-notes-file-of-patient-ash-ketchum',
'https://www.mcsweeneys.net/articles/tinder-is-the-night-and-the-day-and-the-morning-and-the-bathroom',
'https://www.mcsweeneys.net/articles/i-am-an-animator',
'https://www.mcsweeneys.net/articles/that-time-i-published-a-personal-essay-on-the-internet',
'https://www.mcsweeneys.net/articles/an-open-letter-to-preeclampsia',
'https://www.mcsweeneys.net/articles/were-looking-for-a-pop-culture-obsessed-blogger-demon-hunter-to-join-our-team',
'https://www.mcsweeneys.net/articles/9-inappropriate-crushes-you-have-definitely-had-if-your-name-is-sarah-peebles-in-chronological-order',
'https://www.mcsweeneys.net/articles/so-you-want-to-ride-on-my-party-boat',
'https://www.mcsweeneys.net/articles/officer-anthony-engages-the-suspect',
'https://www.mcsweeneys.net/articles/military-id-tag-and-blood-type-tag',
'https://www.mcsweeneys.net/articles/a-poem-about-your-universitys-brand-new-institute',
'https://www.mcsweeneys.net/articles/elizabeth-barrett-browning-how-can-we-love-thee-so-youre-not-so-pissed-off',
'https://www.mcsweeneys.net/articles/other-types-of-tiny-houses',
'https://www.mcsweeneys.net/articles/will-someone-please-tell-me-what-to-do-with-my-body',
'https://www.mcsweeneys.net/articles/an-open-letter-to-lululemon-yoga-pants',
'https://www.mcsweeneys.net/articles/intelligent-design-lacked-adequate-peer-review',
'https://www.mcsweeneys.net/articles/an-exclusive-excerpt-from-harry-potter-and-the-cursing-child',
'https://www.mcsweeneys.net/articles/what-i-wish-for-my-newborn-baby',
'https://www.mcsweeneys.net/articles/objection-your-honor-its-time-for-my-moving-heartfelt-closing-speech-in-this-movie-about-a-trial',
'https://www.mcsweeneys.net/articles/new-england-patriots-key-ring',
'https://www.mcsweeneys.net/articles/the-crippling-anxiety-on-the-west-coast-is-just-as-good-as-the-crippling-anxiety-in-new-york',
'https://www.mcsweeneys.net/articles/i-am-a-bar-mitzvah-entertainer',
'https://www.mcsweeneys.net/articles/welcome-to-pint-sized-the-daycare-bar-for-adults',
'https://www.mcsweeneys.net/articles/i-am-morgma-the-mauler-lord-of-your-local-apartment-rentals',
'https://www.mcsweeneys.net/articles/steve-saves-greg',
'https://www.mcsweeneys.net/articles/i-love-you-but-our-happiness-doesnt-fit-my-personal-brands-narrative-strategy']

mod_text = pd.DataFrame(columns=['article'])

len(urls5)

for url in urls5:
    r = requests.get(url)
    html = r.text
    soup = BeautifulSoup(html, 'html.parser')
    for art in soup.find_all(name='div', attrs={'class':'articleBody'}):
        c = art.text
        mod_text.loc[len(mod_text)]=[c]
    n = random.randint(0,15)
    time.sleep(n)

mod_text.head()

mod_text['article'] = mod_text['article'].str.replace("\n", "")

mod_text['code'] = 100

mod_text.head()

mod_text.to_csv("mod_text.csv", sep="|", encoding="utf-8")





