import newspaper
from newspaper import news_pool
from newspaper import Article

# Build each newspaper
cnn_paper = newspaper.build('http://cnn.com', memoize_articles=False)
nyt_paper = newspaper.build('http://nytimes.com', memoize_articles=False)
fox_paper = newspaper.build('http://foxnews.com', memoize_articles=False)

# Download all the articles multiple threads at a time
papers = [cnn_paper, nyt_paper, fox_paper]
news_pool.set(papers, threads_per_source=2)
news_pool.join()

# Function to create text files with article data from each source
def mkfile(paper,folder,source):
    for i, article in enumerate(paper.articles):
        article.parse()
        t = article.text.lower().encode('ascii','ignore')
        t.replace('\n',' ')
        f = open(folder+source+str(i)+'.txt','w')
        f.write(t)
        f.close()
    

# gather articles from each news source for article classification
mkfile(fox_paper,'articles/fox/','fox')
mkfile(nyt_paper,'articles/nyt/','nyt')
mkfile(cnn_paper,'articles/cnn/','cnn')

# function to scrape article text given a list of article urls
def getart(links,folder,source):
    for i, link in enumerate(links):
        article = Article(link)
        article.download()
        article.parse()
        t = article.text.lower().encode('ascii','ignore')
        t.replace('\n',' ')
        f = open(folder+source+str(i)+'.txt','w')
        f.write(t)
        f.close()

# article urls for additional sentence data

nyt_url = ['https://www.nytimes.com/2017/12/06/opinion/roy-moore-christians.html?action=click&pgtype=Homepage&clickSource=story-heading&module=opinion-c-col-right-region&region=opinion-c-col-right-region&WT.nav=opinion-c-col-right-region&_r=0',
           'https://www.nytimes.com/2017/12/06/opinion/tucson-police-immigration-jeff-sessions.html?action=click&pgtype=Homepage&clickSource=story-heading&module=opinion-c-col-right-region&region=opinion-c-col-right-region&WT.nav=opinion-c-col-right-region',
           'https://www.nytimes.com/2017/12/06/opinion/trump-foreign-policy-giveaway.html?action=click&pgtype=Homepage&clickSource=story-heading&module=opinion-c-col-left-region&region=opinion-c-col-left-region&WT.nav=opinion-c-col-left-region',
           'https://www.nytimes.com/2017/12/06/opinion/doctor-wheelchair-disability.html?action=click&pgtype=Homepage&clickSource=story-heading&module=opinion-c-col-right-region&region=opinion-c-col-right-region&WT.nav=opinion-c-col-right-region',
           'https://www.nytimes.com/2017/12/06/us/politics/franken-harrassment-resign.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news',
           'https://www.nytimes.com/2017/12/06/us/california-fires.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news',
           'https://www.nytimes.com/2017/12/05/world/middleeast/american-embassy-israel-trump-move.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news&_r=0',
           'https://www.nytimes.com/2017/12/05/opinion/does-president-trump-want-to-negotiate-middle-east-peace.html?action=click&pgtype=Homepage&clickSource=story-heading&module=opinion-c-col-left-region&region=opinion-c-col-left-region&WT.nav=opinion-c-col-left-region',
           'https://www.nytimes.com/2017/12/05/sports/olympics/russia-olympics-longman.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news',
           'https://www.nytimes.com/2017/12/06/us/atlanta-mayor.html?module=WatchingPortal&region=c-column-middle-span-region&pgType=Homepage&action=click&mediaId=thumb_square&state=standard&contentPlacement=4&version=internal&contentCollection=www.nytimes.com&contentId=https%3A%2F%2Fwww.nytimes.com%2F2017%2F12%2F06%2Fus%2Fatlanta-mayor.html&eventName=Watching-article-click']

cnn_url = ['http://www.cnn.com/2017/12/05/politics/atlanta-mayor-keisha-lance-bottoms-mary-norwood/index.html',
           'http://www.cnn.com/2017/12/05/middleeast/trump-jerusalem-explainer-intl/index.html',
           'http://money.cnn.com/2017/12/06/investing/coal-tax-cuts-robert-murray-trump/index.html',
           'http://www.cnn.com/2017/12/05/politics/mike-pence-russia-flynn-mueller/index.html',
           'http://www.cnn.com/2017/12/07/politics/roy-moore-campaign-spokeswoman-janet-porter-anderson-cooper-cnntv/index.html',
           'http://www.cnn.com/2017/12/06/politics/trump-jr-testimony-lawmakers-president-trump-tower-meeting/index.html',
           'http://www.cnn.com/2017/12/06/opinions/chemical-safety-opinion-lautenberg/index.html',
           'http://www.cnn.com/2017/12/06/opinions/jerusalem-trump-campaign-promise-stewart-opinion/index.html',
           'http://www.cnn.com/2017/12/06/opinions/give-dreamers-the-chance-my-grandfather-had-garcetti/index.html',
           'http://www.cnn.com/2017/12/04/opinions/ivanka-trump-women-complicit-opinion-costello/index.html']

fox_url = ['http://www.foxnews.com/politics/2017/12/06/republicans-allege-doj-double-standard-as-mueller-probe-takes-heat.html',
           'http://www.foxnews.com/politics/2017/12/06/hamas-plans-day-rage-in-response-to-trumps-jerusalem-decision.html',
           'http://www.foxnews.com/politics/2017/12/06/turnabout-why-trump-gop-leaders-are-now-backing-roy-moore.html',
           'http://www.foxnews.com/opinion/2017/12/06/john-stossel-why-hate-new-york-times.html',
           'http://www.foxnews.com/opinion/2017/12/05/most-incredible-christmas-mistake-ever-made.html',
           'http://www.foxnews.com/opinion/2017/12/05/sarah-sanders-does-press-prays-and-bakes-pies-when-will-haters-like-chelsea-handler-leave-her-alone.html',
           'http://www.foxnews.com/opinion/2017/12/06/roy-moore-is-going-to-win-his-senate-race-despite-democrats-phony-claim-moral-superiority.html',
           'http://www.foxnews.com/opinion/2017/12/07/liberals-attack-doughnut-shops-good-deed-what-in-sweet-name-santa-claus-is-wrong-them.html',
           'http://www.foxnews.com/politics/2017/12/06/conyers-son-was-accused-domestic-abuse-but-not-charged-report-says.html',
           'http://www.foxnews.com/politics/2017/12/06/trump-travel-ban-gets-second-look-from-judges-who-previously-blocked-it.html']

# scrape additional sentence data to expand IBC dataset
getart(nyt_url,'extras/nyt/','nyt')
getart(cnn_url,'extras/cnn/','cnn')
getart(fox_url,'extras/fox/','fox')

breitbart = ['http://www.breitbart.com/big-government/2017/12/07/disgraced-sen-al-franken-lying-left-wing-women-forced-resign/',
            'http://www.breitbart.com/jerusalem/2017/12/07/palestinians-riot-west-bank-gaza-jerusalem-decision/',
            'http://www.breitbart.com/big-government/2017/12/07/nancy-pelosi-gop-using-national-reciprocity-arm-violent-criminals/',
            'http://www.breitbart.com/big-government/2017/12/07/ex-clinton-spokesman-if-obama-cured-cancer-trump-would-try-to-bring-it-back/',
            'http://www.breitbart.com/big-government/2017/12/07/white-house-unfortunate-that-reps-john-lewis-bennie-thompson-wont-join-president-in-honoring-incredible-sacrifice-of-civil-rights-leaders/',
            'http://www.breitbart.com/big-government/2017/12/07/report-doj-officials-strzok-texts-anti-trump/',
            'http://www.breitbart.com/big-government/2017/12/07/justice-official-demoted-as-house-investigators-subpoena-records-of-his-fusion-gps-meetings/',
            'http://www.breitbart.com/big-government/2017/12/07/ted-cruz-mike-lee-endorse-mike-lee-montana-senate-seat/',
            'http://www.breitbart.com/big-government/2017/12/07/time-mag-alabama-dem-doug-jones-has-uphill-battle-due-to-pro-amnesty-anti-border-wall-positions/',
            'http://www.breitbart.com/national-security/2017/12/07/isna-official-islamic-charter-forerunner/',
            'http://www.breitbart.com/big-government/2017/12/07/sarah-sanders-tells-reporters-exactly-what-its-like-to-be-a-woman-working-for-donald-trump/',
            'http://www.breitbart.com/big-government/2017/12/07/no-mention-of-daca-amnesty-at-donald-trump-meeting-with-chuck-and-nancy/',
            'http://www.breitbart.com/big-government/2017/12/07/world-war-ii-veteran-interrupts-trumps-pearl-harbor-speech-sing-remember-pearl-harbor/',
            'http://www.breitbart.com/big-government/2017/12/07/planned-parenthood-ceo-cecile-richards-mocks-insane-natural-family-planning/',
            'http://www.breitbart.com/big-government/2017/12/07/democrats-drop-amnesty-shutdown-threat/',
            'http://www.breitbart.com/big-government/2017/12/07/sanders-deflects-frankens-resignation-calls-trump-resign/',
            'http://www.breitbart.com/big-government/2017/12/07/hitler-obama-in-chicago-speech-nazi-could-rise-in-america-if-people-dont-pay-attention/',
            'http://www.breitbart.com/big-government/2017/12/07/poll-shows-americans-shockingly-misinformed-about-gop-tax-reform/',
            'http://www.breitbart.com/big-government/2017/12/07/national-reciprocity-hinges-on-mitch-mcconnell/',
            'http://www.breitbart.com/big-government/2017/12/07/gohmert-fix-nics-gun-control-expansion-cannot-support/',
            'http://www.breitbart.com/big-hollywood/2017/12/07/naacp-trumps-planned-visit-mississippi-civil-rights-museum-opening/',
            'http://www.breitbart.com/big-government/2017/12/07/wsj-editorial-board-calls-robert-mueller-step-fbi-agents-anti-trump-texts/',
            'http://www.breitbart.com/national-security/2017/12/07/state-department-islamic-state-building-up-north-africa-fails-iraq-syria/',
            'http://www.breitbart.com/jerusalem/2017/12/07/irans-supreme-leader-calls-muslims-unite-major-plot-jerusalem/',
            'http://www.breitbart.com/jerusalem/2017/12/07/israeli-lawmaker-uk-labour-leadership-tainted-anti-semitism/',
            'http://www.breitbart.com/london/2017/12/07/austrian-cardinal-court-ruling-allowing-gay-marriage-denies-reality/',
            'http://www.breitbart.com/jerusalem/2017/12/07/tillerson-good-opportunity-achieve-mideast-peace/',
            'http://www.breitbart.com/big-government/2017/12/07/flotus-and-second-lady-help-at-food-bank-in-texas-visit-with-hurricane-recovery-first-responders/',
            'http://www.breitbart.com/big-government/2017/12/06/bush-bureaucrats-favored-by-john-kelly-now-running-homeland-security-under-trump/',
            'http://www.breitbart.com/big-government/2017/12/06/homeless-veteran-received-400000-woman-helped-buys-home/',
            'http://www.breitbart.com/big-government/2017/12/06/parents-mobilize-to-halt-rule-allowing-k-12-children-to-self-identify-gender-and-race/',
            'http://www.breitbart.com/big-government/2017/12/06/booker-bama-new-jersey-dem-senator-campaign-doug-jones/',
            'http://www.breitbart.com/big-government/2017/12/06/house-judiciary-republicans-call-on-fbi-to-explain-special-status-for-clinton-email-probe/',
            'http://www.breitbart.com/big-government/2017/12/06/jeb-hensarling-fannie-and-freddie-must-be-wound-down/',
            'http://www.breitbart.com/big-government/2017/12/06/roy-moore-campaign-asks-doug-jones-if-he-continues-to-support-obamacare/',
            'http://www.breitbart.com/big-government/2017/12/06/erik-prince-blasts-obama-administration-illegal-surveillance/',
            'http://www.breitbart.com/big-government/2017/12/06/nra-jumps-against-democrat-doug-jones/',
            'http://www.breitbart.com/big-journalism/2017/12/06/ann-coulter-jerry-seinfeld-endorses-roy-moore/']

fox = ['http://www.foxnews.com/opinion/2017/12/07/roger-goodell-s-nfl-contract-is-slap-in-face-to-all-patriotic-americans.html',
      'http://www.foxnews.com/opinion/2017/12/07/democrats-forced-franken-out-to-ramp-up-their-strategy-to-bring-down-trump.html',
      'http://www.foxnews.com/opinion/2017/12/07/judith-miller-jerusalem-now-and-eternal.html',
      'http://www.foxnews.com/opinion/2017/12/07/michael-goodwin-trump-did-right-thing-on-jerusalem.html',
      'http://www.foxnews.com/opinion/2017/12/07/its-christmas-but-grinch-is-alive-and-well-at-this-north-carolina-senior-living-community.html',
      'http://www.foxnews.com/opinion/2017/12/07/newt-gingrich-trumps-monuments-move-wont-harm-environment-ignore-naysayers.html',
      'http://www.foxnews.com/opinion/2017/12/07/why-trump-could-still-pull-fast-one-on-chuck-and-nancy-on-immigration.html',
      'http://www.foxnews.com/opinion/2017/12/07/north-korea-could-launch-its-own-nuclear-pearl-harbor-attacking-us-potentially-killing-millions.html',
      'http://www.foxnews.com/opinion/2017/12/07/liberals-attack-doughnut-shops-good-deed-what-in-sweet-name-santa-claus-is-wrong-them.html',
      'http://www.foxnews.com/opinion/2017/12/07/judge-andrew-napolitano-general-and-president.html',
      'http://www.foxnews.com/opinion/2017/12/06/roy-moore-is-going-to-win-his-senate-race-despite-democrats-phony-claim-moral-superiority.html',
      'http://www.foxnews.com/opinion/2017/12/06/how-insider-trading-scandal-became-insider-leak-scandal-somethings-rotten-in-justice-department.html',
      'http://www.foxnews.com/opinion/2017/12/06/reps-gaetz-biggs-hillary-clintons-fbi-special-treatment-must-be-investigated.html',
      'http://www.foxnews.com/opinion/2017/12/06/heres-how-halifax-explosion-brought-us-and-canada-together-100-years-ago.html',
      'http://www.foxnews.com/opinion/2017/12/06/trump-is-right-israels-capital-is-jerusalem.html',
      'http://www.foxnews.com/opinion/2017/12/06/basket-gun-safety-regulations-means-nothing-if-bureaucrats-arent-held-unaccountable.html',
      'http://www.foxnews.com/opinion/2017/12/06/islamic-terrorists-arent-only-people-out-to-get-theresa-may-right-now.html',
      'http://www.foxnews.com/opinion/2017/12/06/ohio-police-chief-concealed-carry-laws-are-wildly-confusing-here-s-why-need-to-change.html',
      'http://www.foxnews.com/opinion/2017/12/06/matt-lauer-was-fired-week-ago-for-appalling-behavior-not-toxic-masculinity.html',
      'http://www.foxnews.com/opinion/2017/12/05/lewandowski-bossie-trump-is-unique-american-story-that-has-made-our-political-elite-class-irrelevant.html',
      'http://www.foxnews.com/opinion/2017/12/05/for-cybersecurity-it-s-business-not-government-that-should-take-lead-to-protect-private-information.html',
      'http://www.foxnews.com/opinion/2017/12/05/sarah-sanders-does-press-prays-and-bakes-pies-when-will-haters-like-chelsea-handler-leave-her-alone.html',
      'http://www.foxnews.com/opinion/2017/12/05/gregg-jarrett-how-fbi-official-with-political-agenda-corrupted-both-mueller-comey-investigations.html',
      'http://www.foxnews.com/opinion/2017/12/05/most-incredible-christmas-mistake-ever-made.html',
      'http://www.foxnews.com/opinion/2017/12/05/change-in-middle-east-change-must-come-from-people-in-region-thats-why-support-new-saudi-moves.html',
      'http://www.foxnews.com/opinion/2017/12/05/ice-founder-build-wall-now-even-if-government-shutdown-is-price.html',
      'http://www.foxnews.com/opinion/2017/12/05/log-cabin-republican-gay-wedding-cake-case-doesnt-have-to-be-zero-sum-game.html',
      'http://www.foxnews.com/opinion/2017/12/05/when-will-republicans-in-congress-step-up-and-start-restraining-trump.html',
      'http://www.foxnews.com/opinion/2017/12/04/senator-mitch-mcconnell-tax-reform-whats-in-it-for.html',
      'http://www.foxnews.com/opinion/2017/12/04/alan-dershowitz-why-did-flynn-lie-and-why-did-mueller-charge-him-with-lying.html',
      'http://www.foxnews.com/opinion/2017/12/04/why-is-tax-reform-punishing-middle-income-investors-and-retirees.html',
      'http://www.foxnews.com/opinion/2017/12/04/national-park-lovers-should-applaud-trumps-monument-decision.html',
      'http://www.foxnews.com/opinion/2017/12/04/michael-goodwin-left-and-much-washington-are-preparing-to-dance-on-grave-trump-s-presidency.html',
      'http://www.foxnews.com/opinion/2017/12/03/on-international-day-persons-with-disabilities-heres-story-remarkable-boy-in-haiti.html',
      'http://www.foxnews.com/opinion/2017/12/03/despite-nfl-s-anthem-protests-americans-can-meet-again-on-common-ground-to-support-our-nation-s-veterans.html',
      'http://www.foxnews.com/opinion/2017/12/03/new-documents-reveal-fbis-clinton-cover-up.html',
      'http://www.foxnews.com/opinion/2017/12/03/kate-steinles-tragic-death-shows-why-sanctuary-cities-movement-threatens-safety-all-americans.html',
      'http://www.foxnews.com/opinion/2017/12/02/newt-gingrich-congratulations-senate-republicans-on-tax-bill-now-on-to-conference.html',
      'http://www.foxnews.com/opinion/2017/12/02/medias-flynn-sanity-diplomacy-after-election-is-not-same-as-collusion-before-election.html']

nyt = ['https://www.nytimes.com/2017/12/07/opinion/trump-jerusalem-embassy-israel.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=1&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=556536BE4C4355C98B8F1339341B9EF6&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/07/opinion/al-franken-harassment.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=2&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=CD2BD4F5105F734B3864D725150D1221&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/07/opinion/statues-historical-figures.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=3&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=38CD306D7190DC699851836DC76CC974&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/07/opinion/jerusalem-trump.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=4&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=7B12AAECF7D5407C83A4EB43D65BDBF9&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/07/opinion/trump-peace-middle-east.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=5&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=7B353FB157131F0D9E5F5175BB1808C2&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/07/opinion/harvard-graduate-union.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=6&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=50E743AECB7AC1D4E56F78A174615087&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/07/opinion/abortion-supreme-court-speech-california.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=7&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=89CAC9353287B633D794680D0D002D94&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/07/opinion/the-metoo-stories-were-not-hearing.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=8&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=AA46A74C5AF8242B53146B99DF92245C&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/06/opinion/tucson-police-immigration-jeff-sessions.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=9&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=62545DA50BB8B299C8A0C2D5EA423A65&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/06/opinion/roy-moore-christians.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=10&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=2ACA409A7C19EBB2188B52CD7C0CB87A&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/06/opinion/james-levine.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=2&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=DA61D1CFC63D4F1143BE66EE2B148EEA&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/06/opinion/jerusalem-trump.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=3&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=DB2AEC9256DDA928C0E4CF907AB1CAEA&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/06/opinion/facebook-middle-school.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=4&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=01C4979FC4AE6C24E4DA4902FA4840A8&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/06/opinion/trump-populist-agenda.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=5&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=8C34CFAC987033617630EF04F54ADC44&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/06/opinion/clarifying-what-makes-a-diamond.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=6&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=D4999B32D9D94D9E6A77ED29A7DF2724&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/06/opinion/is-media-driving-americans-apart.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=8&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=4D9DB8EEF588E94AA882D6CC0F3D94E5&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/06/opinion/republican-tax-plan.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=9&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=F2A158DA707CB2A9F9841ADEB0389E8C&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/06/opinion/republican-tax-bill.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=17&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=6C3758E1A83506839C7BBE920848B942&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/05/opinion/concealed-gun-laws-national.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=27&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=425722F656792EF281654BE86DE75BE0&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/05/opinion/utah-land-conservation.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=30&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=9D2843BEBC6DAFE75AE2D36F2B5E502A&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/05/opinion/prosecuting-president.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=31&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=7D9579021FFA116616E04AA8CE620264&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/05/opinion/net-neutrality.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=40&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=2E05B51CD975B8447EF5677BAA9664A6&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/05/opinion/adapt-climate-change.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=43&pgtype=sectionfront',
      'https://www.nytimes.com/2017/12/05/opinion/george-mcgovern-vietnam-democratic.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=45&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=520DEBA77C91A79E03EF4C5CC97F19EF&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/05/opinion/is-trump-crazy-like-a-fox-or-plain-old-crazy.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=47&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=C1ABB59378D9A6303AD03D369E5AF401&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/05/opinion/trump-mueller-facts.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=49&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=C272CECACD62A355AC137BDCF163F530&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/05/opinion/xi-jinping-china-rises.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=53&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=8D565934F04ED2D975462992EE703BAB&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/04/opinion/millennials-hate-capitalism.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=57&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=C8F90FB91428BC7F9A42C9FDDF2CE9E2&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/04/opinion/gay-marriage-cake-case.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=58&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=C2752C4234A40E642834941849482485&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/04/opinion/trump-impeach-constitution.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=59&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=96EA886FB0C3550AB579D0F8ED0F23F7&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/04/opinion/net-neutrality-overblown-concerns.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=60&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=A5A52A8222BA248D8BBD2D5B322BB986&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/section/opinion?action=click&pgtype=Homepage&region=TopBar&module=HPMiniNav&contentCollection=Opinion&WT.nav=page',
      'https://www.nytimes.com/2017/12/04/opinion/president-trump-obstruction-justice.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=62&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=B3F714122CD23DD9E99B3A3FBC362DDA&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/04/opinion/tax-bill.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=65&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=96A3F6846CF20C7A789A8E35F5EAB22A&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/04/opinion/trump-michael-flynn-billy-bush.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=67&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=BFEBA5D4769C28AFEBAA5D4D5E11BBEF&gwt=pay&assetType=opinion',
      'https://krugman.blogs.nytimes.com/2017/12/04/leprechauns-of-eastern-europe/?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=68&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=10B71336BC4CE51B54664D965843C767&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/04/opinion/first-amendment-wedding-cake.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=72&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=734FB5D901A977CC97EC7804F95E04DE&gwt=pay&assetType=opinion',
      'https://www.nytimes.com/2017/12/04/opinion/trump-flynn-fbi.html?rref=collection%2Fsectioncollection%2Fopinion&action=click&contentCollection=opinion&region=stream&module=stream_unit&version=latest&contentPlacement=76&pgtype=sectionfront&mtrref=www.nytimes.com&gwh=C7DF8AF4F36395E0C2C51A23821579E9&gwt=pay&assetType=opinion']

new_yorker = ['https://www.newyorker.com/news/news-desk/palestinians-reject-trumps-jerusalem-declaration',
             'https://www.newyorker.com/news/our-columnists/al-franken-resignation-and-the-selective-force-of-metoo',
             'https://www.newyorker.com/news/news-desk/in-alabama-roy-moores-supporters-rail-at-establishment-republicans',
             'https://www.newyorker.com/news/news-desk/would-trump-consider-a-court-packing-scheme',
             'https://www.newyorker.com/news/news-desk/republican-women-in-alabama-sound-off-on-moore',
             'https://www.newyorker.com/news/news-desk/even-while-pleading-guilty-michael-flynn-invokes-his-idea-of-service',
             'https://www.newyorker.com/news/news-desk/how-rex-tillerson-wrecked-the-state-department',
             'https://www.newyorker.com/news/news-desk/trump-britain-first-and-the-purveyors-of-racism-and-bigotry',
             'https://www.newyorker.com/sections/news/can-time-inc-survive-the-kochs',
             'https://www.newyorker.com/news/news-desk/what-to-do-with-monuments-whose-history-weve-forgotten',
             'https://www.newyorker.com/news/news-desk/the-republican-tax-plan-contains-more-middle-class-pain-than-even-its-critics-are-saying',
             'https://www.newyorker.com/news/news-desk/the-catastrophe-of-saudi-arabias-trump-backed-intervention-in-yemen',
             'https://www.newyorker.com/news/news-desk/the-shocking-math-of-the-republican-tax-plan',
             'https://www.newyorker.com/news/our-columnists/the-gops-boil-the-frog-strategy-to-save-trump',
             'https://www.newyorker.com/sections/news/karl-rove-has-seen-the-enemy-and-he-is-steve-bannon',
             'https://www.newyorker.com/news/amy-davidson-sorkin/its-all-connected-jeff-sessions-roy-moore-and-a-new-russia-investigation',
             'https://www.newyorker.com/news/news-desk/how-republicans-aim-to-combine-the-destruction-of-obamacare-with-a-tax-cut-for-the-rich',
             'https://www.newyorker.com/news/news-desk/eric-holders-battle-against-gerrymandering',
             'https://www.newyorker.com/news/news-desk/russias-election-meddling-is-another-american-intelligence-failure',
             'https://www.newyorker.com/news/news-desk/how-trump-is-quietly-dismantling-the-architecture-of-global-governance',
             'https://www.newyorker.com/news/news-desk/trump-official-behind-the-end-of-daca-explains-himself',
             'https://www.newyorker.com/news/news-desk/trumpism-stumbles-in-virginia-and-republicans-fall-to-a-democratic-wave',
             'https://www.newyorker.com/news/news-desk/how-the-gang-ms-13-became-a-trumpian-campaign-issue-in-virginia',
             'https://www.newyorker.com/news/news-desk/the-fate-of-populism-in-2018',
             'https://www.newyorker.com/news/news-desk/john-kellys-bizarre-mythology-of-the-civil-war',
             'https://www.newyorker.com/news/news-desk/the-democratic-civil-war-is-getting-nasty-even-if-no-one-is-paying-attention',
             'https://www.newyorker.com/news/news-desk/paul-manafort-and-the-case-of-the-250000-antique-rug-store-bill',
             'https://www.newyorker.com/news/news-desk/the-ignorance-of-trumps-vague-tax-plan',
             'https://www.newyorker.com/news/news-desk/the-trump-officials-making-abortion-an-issue-at-the-uss-refugee-office',
             'https://www.newyorker.com/news/news-desk/why-roy-moores-law-school-professor-nicknamed-him-fruit-salad',
             'https://www.newyorker.com/news/john-cassidy/what-myeshia-johnson-revealed-about-donald-trump',
             'https://www.newyorker.com/news/ryan-lizza/john-kelly-and-the-dangerous-moral-calculus-of-working-for-trump',
             'https://www.newyorker.com/news/news-desk/the-iran-business-ties-trump-didnt-disclose',
             'https://www.newyorker.com/news/benjamin-wallace-wells/what-democrats-are-fighting-about-in-california',
             'https://www.newyorker.com/news/news-desk/journalisms-broken-business-model-wont-be-solved-by-billionaires',
             'https://www.newyorker.com/news/ryan-lizza/how-trump-is-empowering-the-democrats',
             'https://www.newyorker.com/news/john-cassidy/waiting-for-the-trump-slump-in-the-stock-market',
             'https://www.newyorker.com/news/amy-davidson-sorkin/donald-trumps-unseemly-condolence-call-bragging-game',
             'https://www.newyorker.com/news/john-cassidy/how-far-will-john-mccain-go-against-president-trump']

cnn = ['http://www.cnn.com/2017/12/05/opinions/gop-tax-plan-income-inequality-second-gilded-age-ellis-opinion/index.html',
      'http://www.cnn.com/2017/12/05/opinions/access-hollywood-arianne-zucker-opinion/index.html',
      'http://www.cnn.com/2017/12/05/opinions/why-trump-is-winning-bauerlein/index.html',
      'http://www.cnn.com/2017/12/05/opinions/roy-moore-john-conyers-congress-filipovic-opinion/index.html',
      'http://www.cnn.com/2017/12/05/opinions/dodd-frank-rollbacks-opinion-brown/index.html',
      'http://www.cnn.com/2017/12/02/opinions/trump-family-wont-go-to-jail-pate-opinion/index.html',
      'http://www.cnn.com/2017/12/04/opinions/obama-tried-to-save-trump-from-mistake-opinion-dantonio/index.html',
      'http://www.cnn.com/2017/12/04/opinions/trump-dershowitz-obstruction-of-justice-callan-opinion/index.html',
      'http://www.cnn.com/2017/12/04/opinions/ivanka-trump-women-complicit-opinion-costello/index.html',
      'http://www.cnn.com/2017/12/04/opinions/republican-tax-plan-trump-opinion-kohn/index.html',
      'http://www.cnn.com/2017/12/03/opinions/mueller-investigation-sacrificial-lamb-opinion-gagliano/index.html',
      'http://www.cnn.com/2017/12/03/opinions/jerusalem-capital-trump-opinion-miller/index.html',
      'http://www.cnn.com/2017/11/26/opinions/ivanka-tillerson-womens-rights-opinion-hossain/index.html',
      'http://www.cnn.com/2017/11/16/opinions/sexual-harassment-party-politics-roxanne-jones-opinion/index.html',
      'http://www.cnn.com/2017/11/10/opinions/roy-moore-opinion-robbins/index.html',
      'http://www.cnn.com/2017/10/13/opinions/is-it-time-to-talk-the-25th-amendment-opinion-zelizer/index.html',
      'http://www.cnn.com/2017/11/07/opinions/randazza-even-trump-has-a-right-to-free-speech-opinion/index.html',
      'http://www.cnn.com/2017/11/06/opinions/east-asia-stuck-cold-war-hotta/index.html',
      'http://www.cnn.com/2017/11/01/opinions/the-trump-campaigns-cocktail-of-stupid-jennings/index.html',
      'http://www.cnn.com/2017/12/04/opinions/masterpiece-cakeshop-colorado-opinion-grimm/index.html',
      'http://www.cnn.com/2017/12/01/opinions/trump-hiv-aids-budget-cuts-hart-opinion/index.html',
      'http://www.cnn.com/2017/12/01/opinions/sexual-harassment-reconciliation-opinion-godfrey-ryan/index.html',
      'http://www.cnn.com/2017/11/30/opinions/women-complicit-harassment-lauer-roxanne-jones-opinion/index.html',
      'http://www.cnn.com/2017/12/07/opinions/how-democrats-win-the-future-perez-ellison/index.html',
      'http://www.cnn.com/2017/12/06/opinions/trump-global-trust-lost-opinion-ghitis/index.html',
      'http://www.cnn.com/2017/12/05/opinions/roy-moore-john-conyers-congress-filipovic-opinion/index.html',
      'http://www.cnn.com/2017/12/05/opinions/dodd-frank-rollbacks-opinion-brown/index.html',
      'http://www.cnn.com/2017/12/04/opinions/obama-tried-to-save-trump-from-mistake-opinion-dantonio/index.html',
      'http://www.cnn.com/2017/12/04/opinions/republican-tax-plan-trump-opinion-kohn/index.html',
      'http://www.cnn.com/2017/12/02/opinions/gop-tax-plan-brainless-sachs-opinion/index.html',
      'http://www.cnn.com/2017/12/02/opinions/trump-dual-track-presidency-zelizer-opinion/index.html',
      'http://www.cnn.com/2017/12/01/opinions/trump-team-and-russians-flynn-opinion-callan/index.html',
      'http://www.cnn.com/2017/12/01/opinions/its-time-to-start-talking-about-impeachment-louis/index.html',
      'http://www.cnn.com/2017/11/29/opinions/gop-tax-plan-women-families-rowe-finkbeiner-opinion/index.html',
      'http://www.cnn.com/2017/11/29/opinions/press-freedom-for-student-journalists-lomonte-opinion/index.html',
      'http://www.cnn.com/2017/11/28/opinions/trump-cfpb-this-is-not-normal-hochberg-opinion/index.html',
      'http://www.cnn.com/2017/11/28/opinions/what-happens-when-media-gets-it-really-wrong-campbell-free-press/index.html',
      'http://www.cnn.com/2017/11/26/opinions/veterans-ptsd-disrcharge-opinion-heffinger/index.html']

# Scrape opinion and political artices for these 5 major news sources
#(used for training RNN on articles as a whole rather than individual sentences)
getart(breitbart,'polit/breit/','breit')
getart(fox,'polit/fox/','fox')
getart(nyt,'polit/nyt/','nyt')
getart(new_yorker,'polit/new_yorker/','new_yorker')
getart(cnn,'polit/cnn/','cnn')

getart(cnn,'polit/cnn/','cnn')



