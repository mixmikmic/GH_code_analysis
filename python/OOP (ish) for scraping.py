from bs4 import BeautifulSoup as bf
import requests
import re
import pickle

def load_soup(url):
    html = requests.get(url).text
    soup = bf(html, 'html5lib')
    return soup

# builds url lists specific to the site
def Drugs_url_list(drug_stem, pg_init, pg_n):
    url_c = ['https://www.drugs.com/comments/', '/?page=']
    url_list = [url_c[0] + drug_stem + url_c[1] +str(ik) for ik in range(pg_init, pg_n+1)]
    return url_list

def WebMD_url_list(drug_stem, pg_init, pg_n):
    url_c = ['https://www.webmd.com/drugs/drugreview-', '&pageIndex=', '&sortby=3&conditionFilter=-500' ]
    url_list = [url_c[0] + drug_stem + url_c[1] +str(ik) + url_c[2] for ik in range(pg_init, pg_n+1)]
    return url_list
    

# takes url list, scrapes, returns pages of soup
def scraper(url_list):
    soup_list = []
    for url in url_list:
        soup_list.append( [url, load_soup(url)])
    return soup_list 

# uses methods from the two parser classes to slice and dice soup into review objects
def parse_reviews(pages, tag, drug, parser):
#     print(len(pages))
    for page in pages:
        rev_stew = page[1].find_all('div' ,{'class' : tag})
        for ik, item in enumerate(rev_stew):
            new_review = review(drug, item, parser, ik)
            drug.reviews.append(new_review)

# each drug has a name and a url_stem
class drug:
    
    def __init__(self, name, url_stem):
        self.name = name
        self.url_stem = url_stem
        self.reviews = []

# ************ Drugs.com ************  Parser

# modified from source: 
# https://blog.nycdatascience.com/student-works/web-scraping/anti-epileptic-drug-review-analysis/

class DrugsDotCom:
    
        def __init__(self, name):
            self.name = name

        # fetch information about author;
        # bug fix: added a tag to the tags list.  There may be more lurking...
        def set_reviewerMeta (self, _rev_soup, ik):
            tags = ['user-name user-type user-type-2_non_member', 'user-name user-type user-type-1_standard_member','user-name user-type user-type-0_select_member']
            if _rev_soup.find('p', {'class': tags[0]}):
                return _rev_soup.find('p', {'class': tags[0]})
            elif _rev_soup.find('p', {'class': tags[1]}):
                return _rev_soup.find('p', {'class': tags[1]})
            elif _rev_soup.find('p', {'class': tags[2]}):
                return _rev_soup.find('p', {'class': tags[2]})
            else:
                return None
#                         print(tag)
#                         return _rev_soup.find('p', {'class': tag}) 
#                     except:
#                         print(MetaData)
#                         MetaData = 2
#                 return MetaData
#                 for tag in ['user-name user-type user-type-2_non_member', 'user-name user-type user-type-0_select_member']:
#                     try:
#                         print(tag)
#                         return _rev_soup.find('p', {'class': tag}) 
#                     except:
#                         print(MetaData)
#                         MetaData = 2
#                 return MetaData
#                 if MetaData != 2:
#                     print(MetaData)
#                     return MetaData
#                 else:
#                     return 2

            
        def set_userName (self, _reviewerMeta):
                try:
#                     print(_reviewerMeta.contents[0])
#                     print(_reviewerMeta.b.get_text())
#                     return _reviewerMeta.b.get_text()
                    return _reviewerMeta.contents[0].strip()
                except:
                    return 'Anonymous'

                
        #need to fix this        
        def set_ageRange (self, _reviewerMeta):
                    try:
                        return re.search('\s\w+[-]\w+\s', _reviewerMeta).group().strip()
                    except:
                        return None
                    
                    
        #gender not specified on drugs.com
        def set_gender (self, _reviewerMeta):
                return None
            
        #role not specified on drugs.com
        def set_role(self, _rev_soup):
            return None
            
        def set_medDuration (self, _reviewerMeta):
                try: 
#                     print('med duration: ', _reviewerMeta.find_all('span', {'class':'small light'})[0].text.replace('(', '').replace(')', '').replace('taken for ', ''))
#                     print(reviewerDates[0])
#                     print('med duration: ', reviewerDates[0].contents[0].replace('(', '').replace(')', '').replace('taken for ', ''))
                    return _reviewerMeta.find_all('span', {'class':'small light'})[0].text.replace('(', '').replace(')', '').replace('taken for ', '')
#                     return _reviewerMeta.find('span', {'class': 'small light'}).text()
                except:
                    return None

                
        def set_reviewDate (self, _reviewerMeta):
                try:
                    print('review date: ' ,_reviewerMeta.find_all('span', {'class':'light comment-date'})[0].text)
                    return _reviewerMeta.find_all('span', {'class':'light comment-date'})[0].text
#                     return _reviewerMeta.find('span', {'class': 'light comment-date'}).text()
                except:
                    return None
                
                
        def set_condition (self, _rev_soup):
                try:
                    return _rev_soup.find('div', {'class':'user-comment'}).b.get_text()
                except:
                    return None
                
                
        #from WebMD.com
        def set_effectiveness (self, _rev_soup):
                return None
            
            
        #from WebMD.com
        def set_ease_of_use (self, _rev_soup):
                return None
            
            
        #from WebMD.com
        def set_satisfaction (self, _rev_soup):
                return None

            
        def set_genRating (self, _rev_soup):        
                try:
                    return _rev_soup.find('div',{'class': 'rating-score'}).get_text()
                except:
                    return None


        def set_comment (self, _rev_soup, ik):
                try:
                    return _rev_soup.find('div', {'class':'user-comment'}).span.get_text()
                except:
                    return None


        def set_upVotes (self, _rev_soup):
                try: 
                    temp = _rev_soup.find('p', {'class': 'note'}).b.get_text()
                    return int(re.search(r'\d+', temp).group())
                except:
                    return None

# ************ webMD.com ************ Parser

class WebMD:
    
        def __init__(self, name):
            self.name = name

            
        def set_reviewerMeta (self, _rev_soup, ik):
            try:
                return _rev_soup.find('p', {'class':'reviewerInfo'}).text.strip('Reviewer: ')
            except:
#                 print(_rev_soup)
                return None
            
        #below takes reviewer soup
        def set_userName (self, _reviewerMeta):
            try:
                splits = _reviewerMeta.split(',')
                if len(splits)>1:
                    return splits[0]
                else:
                    return 'Anonymous'
            except:
                return 'Anonymous'

        def set_ageRange (self, _reviewerMeta):
            try:
                return re.search('\s\w+[-]\w+\s', _reviewerMeta).group().strip()
            except:
                return None

        def set_gender (self, _reviewerMeta):
            try: 
                return re.split('\s\w+[-]\w+\s', _reviewerMeta)[1].split()[0]
            except: 
                return None
        
        def set_role(self, _rev_soup):
            try:
                return _rev_soup.find('p', {'class':'reviewerInfo'}).text.strip('Reviewer: ').split(' ')[-1].replace('(','').replace(')','')
            except:
                return None
            
        def set_medDuration (self, _reviewerMeta):
            try:
                return re.split('on Treatment for ', _reviewerMeta)[1].split('(Patient)')[0].strip()
            except:
                return None
            
        #below takes full soup
        #untested for webMD
        def set_reviewDate  (self, _rev_soup):
            try:
                return _rev_soup.find('div', {'class': 'date'}).text.split(' ',1)[0]

            except:
                return None
                
        def set_condition (self, _rev_soup):
            try:
                condition = _rev_soup.find('div', {'class': 'conditionInfo'}).text
                temp = condition.split('Condition: ')[1]
                return temp
            except:
                return None
                
        def set_effectiveness (self, _rev_soup):
                try:
                    temp = _rev_soup.find('div' ,{'class' : 'catRatings firstEl clearfix'}).text
                    return int(re.search(r'\d+', temp).group())
                except:
                    return None

        def set_ease_of_use (self, _rev_soup):
                try:
                    temp = _rev_soup.find('div' ,{'class' : 'catRatings clearfix'}).text
                    return int(re.search(r'\d+', temp).group())
                except:
                    return None

        def set_satisfaction (self, _rev_soup):
                try:
                    temp = _rev_soup.find('div' ,{'class' : 'catRatings lastEl clearfix'}).text
                    return int(re.search(r'\d+', temp).group())
                except:
                    return None
                
        #from drugs.com        
        def set_genRating  (self, _rev_soup):
                    return None

        def set_comment (self, _rev_soup, ik):
                try: 
                    temp = _rev_soup.find('p', {'id':'comFull'+str(ik+1)}).text
                    temp = re.split('Hide Full', temp)[0]
                    return temp.lstrip('Comment:')
                except:
                    return None

        def set_upVotes (self, _rev_soup):
                try:
                    temp = _rev_soup.find('p', {'class' : "helpful"}).text
                    return int(re.search(r'\d+', temp).group())
                except:
                    return None

# Review object
class review(drug):
    
        def __init__(self, drug, _review_soup, site, ik):
            
            reviewer_info = site.set_reviewerMeta(_review_soup, ik)
#             reviewer_dates = site.set_reviewerDates(reviewer_info)
#             print(reviewer_info)
            self.drugName = drug.name
            self.site = site.name
            self.condition = site.set_condition(_review_soup)
            self.reviewDate = site.set_reviewDate(_review_soup)
            self.userName = site.set_userName(reviewer_info) #temp.split(',')[0]
#             print(self.reviewDate)
            self.ageRange = site.set_ageRange(reviewer_info) #re.search('\s\w+[-]\w+\s', temp).group().strip()
            self.gender = site.set_gender(reviewer_info) #re.split('\s\w+[-]\w+\s', temp)[1].split()[0]
            self.role = site.set_role(_review_soup)
            self.medDuration = site.set_medDuration(reviewer_info) #re.split('on Treatment for ', temp)[1].split('(Patient)')[0].strip()
            self.effectiveness = site.set_effectiveness(_review_soup)
            self.ease_of_use = site.set_ease_of_use(_review_soup)
            self.satisfaction = site.set_satisfaction(_review_soup)
            self.genRating = site.set_genRating(_review_soup)
            self.comment = site.set_comment(_review_soup, ik)
            self.upVotes = site.set_upVotes(_review_soup)

# Get Abilify Soup
abilify_stem = '64439-Abilify-oral.aspx?drugid=64439&drugname=Abilify-oral'
abilify = drug('abilify', abilify_stem)
abilify_Soup = scraper(WebMD_url_list(abilify.url_stem, 0, 140))

# Parse Abilify Soup
abilify = drug('abilify', abilify_stem)
webMD_parser = WebMD('webMD')
webMD_tag = 'userPost'
parse_reviews(abilify_Soup, webMD_tag, abilify, webMD_parser)

# prints review objects for inspection

for reviewx in abilify.reviews[0:10]:
    for key in reviewx.__dict__:
        print(key, ':', reviewx.__dict__[key])
    print('----------')
    print('')

#trouble shoot patient metadata
abilify_Soup[0][1].find('p', {'class':'reviewerInfo'}).text
rev_stew[1].find('p', {'class':'reviewerInfo'}).text

for page in abilify_Soup:
        rev_stew = page[1].find_all('div' ,{'class' : webMD_tag})
        for ik, item in enumerate(rev_stew):
            print(item)
            break

# Drugs.com test

meth_stem = 'methylphenidate'
meth = drug('methylphenidate', meth_stem)
meth_Soup = scraper(Drugs_url_list(meth.url_stem, 0, 2))

meth = drug('methylphenidate', meth_stem)
drugs_tag = 'block-wrap comment-wrap'
drugs_parser = DrugsDotCom('drugsDotCom')
parse_reviews(meth_Soup, drugs_tag, meth, drugs_parser)

# Pickle Soup (at the very least--work on pickling objects (?) later)

dict_list = []
for reviewx in abilify.reviews:
    dict_list.append(reviewx.__dict__)

pickle.dump( dict_list, open( "abilify.p", "wb" ) )



