# Load required libraries
import nltk
import requests
from bs4 import BeautifulSoup
import re
import lxml
import pandas as pd
import numpy as np
from sklearn.externals import joblib

def scrape(hyperlink):
    """Use the Requests library to scrape a Kickstarter project page
    
    Args:
        hyperlink (str): URL of the Kickstarter page to scrape
    
    Returns:
        a response object containing the scraped content"""
    
    return requests.get(hyperlink)

def parse(scraped_html):
    """Use the BeautifulSoup library to parse the scraped HTML of a project 
    using an lxml parser
    
    Args:
        scraped_html (response object): the unparsed response object collected
        by the web scraper
    
    Returns:
        a soup object containing the parsed HTML"""
    
    # Parse the HTML content using an lxml parser
    return BeautifulSoup(scraped_html.text, 'lxml')

def clean_up(messy_text):    
    """Clean up the text of a campaign section by removing unnecessary and
    extraneous content
    
    Args:
        messy_text (str): the raw text from a campaign section
    
    Returns:
        a string containing the cleaned text"""
    
    # Remove line breaks, leading and trailing whitespace, and compress all
    # whitespace to a single space
    clean_text = ' '.join(messy_text.split()).strip()
    
    # Remove the HTML5 warning for videos
    return clean_text.replace(
        "You'll need an HTML5 capable browser to see this content. " + \
        "Play Replay with sound Play with sound 00:00 00:00",
        ''
    )

def get_campaign(soup):
    """Extract the two campaign sections, "About this project" and "Risk and
    challenges", of a Kickstarter project
    
    Args:
        soup (soup object): parsed HTML content of a Kickstarter project page
    
    Returns:
        a dictionary of 2 strings containing each campaign section"""
    
    # Collect the "About this project" section if available
    try:
        section1 = soup.find(
            'div',
            class_='full-description js-full-description responsive-media ' + \
                'formatted-lists'
        ).get_text(' ')
    except AttributeError:
        section1 = 'section_not_found'
    
    # Collect the "Risks and challenges" section if available, and remove all
    # unnecessary text
    try:
        section2 = soup.find(
            'div', 
            class_='mb3 mb10-sm mb3 js-risks'
        ) \
            .get_text(' ') \
            .replace('Risks and challenges', '') \
            .replace('Learn about accountability on Kickstarter', '')
    except AttributeError:
        section2 = 'section_not_found'
    
    # Clean both sections
    return {'about': clean_up(section1), 'risks': clean_up(section2)}

# Select a test hyperlink
hyperlink = 'https://www.kickstarter.com/projects/1385294316/help-me-start' +     '-my-cottage-industry-bakesalecom?ref=category_newest'

# Scrape the project page
scraped_html = scrape(hyperlink)

# Parse the scraped HTML and collect the campaign sections
soup = parse(scraped_html)
campaign = get_campaign(soup)

def normalize(text):
    """Tag meta content inside a campaign section, such as email addresses,
    hyperlinks, money amounts, percentages, phone numbers, and numbers, to
    avoid adding these into the word count
    
    Args:
        text (str): cleaned text of a campaign section
    
    Returns:
        a string containing the text of a campaign section with all the meta
        content tagged"""
    
    # Tag email addresses with regex
    normalized = re.sub(
        r'\b[\w\-.]+?@\w+?\.\w{2,4}\b',
        'emailaddr',
        text
    )
    
    # Tag hyperlinks with regex
    normalized = re.sub(
        r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)',
        'httpaddr',
        normalized
    )
    
    # Tag money amounts with regex
    normalized = re.sub(r'\$\d+(\.\d+)?', 'dollramt', normalized)
    
    # Tag percentages with regex
    normalized = re.sub(r'\d+(\.\d+)?\%', 'percntg', normalized)
    
    # Tag phone numbers with regex
    normalized = re.sub(
        r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phonenumbr',
        normalized
    )
    
    # Tag plain numbers with regex
    return re.sub(r'\d+(\.\d+)?', 'numbr', normalized)

# Normalize campaign sections
campaign['about'] = normalize(campaign['about'])
campaign['risks'] = normalize(campaign['risks'])

def get_sentences(text):
    """Use the sentence tokenizer from the nltk library to extract sentences 
    from the text of a campaign section
    
    Args:
        text (str): cleaned and normalized text of a campaign section
    
    Returns:
        a list containing every sentence of a campaign section"""
    
    # Tokenize the text into sentences
    return nltk.sent_tokenize(text)

# If the campaign section is missing, assign NaN to 'num_sents', otherwise
# count and display the number of sentences
if campaign['about'] == 'section_not_found':
    num_sents = np.nan
else:
    num_sents = len(get_sentences(campaign['about']))
num_sents

def remove_punc(text):
    """Remove all punctuation from the text of a campaign section
    
    Args:
        text (str): cleaned and normalized text of a campaign section
    
    Returns:
        a string containing the text of a campaign section without any
        punctuation"""
    
    # Remove punctuation with regex
    return re.sub(r'[^\w\d\s]|\_', ' ', text)

def get_words(text):
    """Use the word tokenizer from the nltk library to extract words from the 
    text of a campaign section
    
    Args:
        text (str): cleaned and normalized text of a campaign section
    
    Returns:
        a list containing every word of a campaign section"""
    
    # Remove punctuation and then tokenize the text into words
    return [word for word in nltk.word_tokenize(remove_punc(text))]

# If the campaign section is missing, assign NaN to 'num_words', otherwise
# count and display the number of words
if campaign['about'] == 'section_not_found':
    num_words = np.nan
else:
    num_words = len(get_words(campaign['about']))

# If the section contains no words, assign NaN to num_words to catch potential
# division by zero errors
if num_words == 0:
    num_words = np.nan 
num_words

def identify_allcaps(text):
    """Find all examples where a word is written in all capital letters in the 
    text of a campaign section
    
    Args:
        text (str): cleaned and normalized text of a campaign section
    
    Returns:
        a list containing every all-caps word of a campaign section"""
        
    # Identify all-caps words with regex
    return re.findall(r'\b[A-Z]{2,}', text)

# Display 'NaN' if the section isn't found, otherwise display the # of all-caps
# words and its %
if campaign['about'] == 'section_not_found':
    print(np.nan, np.nan)
else:
    print(
        len(identify_allcaps(campaign['about'])),
        len(identify_allcaps(campaign['about'])) / num_words
    )

def count_exclamations(text):
    """Count the number of exclamation marks in the text of a campaign section
    
    Args:
        text (str): cleaned and normalized text of a campaign section
    
    Returns:
        an integer representing the number of exclamation marks in the text"""
    
    # Count the number of exclamation marks in the text
    return text.count('!')

# Display 'NaN' if the section isn't found, otherwise display the # of ! marks
# and its %
if campaign['about'] == 'section_not_found':
    print(np.nan, np.nan)
else:
    print(
        count_exclamations(campaign['about']),
        count_exclamations(campaign['about']) / num_words
    )

def count_apple_words(text):
    """Count the number of innovation-related words (called Apple words) in the
    text of a campaign section
    
    Args:
        text (str): cleaned and normalized text of a campaign section
    
    Returns:
        an integer representing the number of Apple words in the text"""
    
    # Define a set of adjectives used commonly by Apple marketing team
    # according to https://www.youtube.com/watch?v=ZWPqjXYTqYw
    apple_words = frozenset(
        ['revolutionary', 'breakthrough', 'beautiful', 'magical', 
        'gorgeous', 'amazing', 'incredible', 'awesome']
    )
    
    # Count total number of Apple words in the text
    return sum(1 for word in get_words(text) if word in apple_words)

# Display 'NaN' if the section isn't found, otherwise display the # of Apple
# adjectives and its %
if campaign['about'] == 'section_not_found':
    print(np.nan, np.nan)
else:
    print(
        count_apple_words(campaign['about']),
        count_apple_words(campaign['about']) / num_words
    )

def compute_avg_words(text):
    """Count the average number of words in each sentence in the text of a 
    campaign section
    
    Args:
        text (str): cleaned and normalized text of a campaign section
    
    Returns:
        a float representing the average number of words in each sentence of
        the text"""
    
    # Compute the average number of words in each sentence
    return pd.Series(
        [len(get_words(sentence)) for sentence in \
         get_sentences(text)]
    ).mean()

# Display 'NaN' if the section isn't found, otherwise display the average # of
# words per sentence
if campaign['about'] == 'section_not_found':
    print(np.nan)
else:
    print(compute_avg_words(campaign['about']))

def count_paragraphs(soup, section):    
    """Count the number of paragraph tags in a campaign section
    
    Args:
        soup (soup object): parsed HTML content of a Kickstarter project page
        section (str): label of the campaign section to be analyzed
    
    Returns:
        an integer representing the number of paragraphs in the campaign
        section"""
    
    # Use tree parsing to count the number of paragraphs depending on which
    # section is requested
    if section == 'about':
        return len(soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
                '-media formatted-lists'
        ).find_all('p'))
    elif section == 'risks':
        return len(soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('p'))

# Display 'NaN' if the section isn't found, otherwise display the # of
# paragraphs
if campaign['about'] == 'section_not_found':
    print(np.nan)
else:
    print(count_paragraphs(soup, 'about'))

def compute_avg_sents_paragraph(soup, section):
    """Count the average number of sentences per paragraph in a campaign 
    section
    
    Args:
        soup (soup object): parsed HTML content of a Kickstarter project page
        section (str): label of the campaign section to be analyzed
    
    Returns:
        a float representing the average number of sentences in each paragraph
        of a campaign section"""
    
    # Use tree parsing to identify all paragraphs depending on which section
    # is requested
    if section == 'about':
        paragraphs = soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
                '-media formatted-lists'
        ).find_all('p')
    elif section == 'risks':
        paragraphs = soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('p')
    
    # Compute the average number of sentences in each paragraph
    return pd.Series(
        [len(get_sentences(paragraph.get_text(' '))) for paragraph in \
         paragraphs]
    ).mean()

# Display 'NaN' if the section isn't found, otherwise display the average # of
# sentences per paragraph
if campaign['about'] == 'section_not_found':
    print(np.nan)
else:
    print(compute_avg_sents_paragraph(soup, 'about'))

def compute_avg_words_paragraph(soup, section):
    """Count the average number of words per paragraph in a campaign section
    
    Args:
        soup (soup object): parsed HTML content of a Kickstarter project page
        section (str): label of the campaign section to be analyzed
    
    Returns:
        a float representing the average number of words in each paragraph
        of a campaign section"""
    
    # Use tree parsing to identify all paragraphs depending on which section
    # is requested
    if section == 'about':
        paragraphs = soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
                '-media formatted-lists'
        ).find_all('p')
    elif section == 'risks':
        paragraphs = soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('p')
    
    # Compute the average number of words in each paragraph
    return pd.Series(
        [len(get_words(paragraph.get_text(' '))) for paragraph in paragraphs]
    ).mean()

# Display 'NaN' if the section isn't found, otherwise display the average # of
# words per paragraph
if campaign['about'] == 'section_not_found':
    print(np.nan)
else:
    print(compute_avg_words_paragraph(soup, 'about'))

def count_images(soup, section):    
    """Count the number of image tags in a campaign section
    
    Args:
        soup (soup object): parsed HTML content of a Kickstarter project page
        section (str): label of the campaign section to be analyzed
    
    Returns:
        an integer representing the number of images in a campaign section"""
    
    # Use tree parsing to identify all image tags depending on which section
    # is requested
    if section == 'about':
        return len(soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
                '-media formatted-lists'
        ).find_all('img'))
    elif section == 'risks':
        return len(soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('img'))

# Display 'NaN' if the section isn't found, otherwise display the # of images
if campaign['about'] == 'section_not_found':
    print(np.nan)
else:
    print(count_images(soup, 'about'))

def count_videos(soup, section):    
    """Count the number of non-YouTube videos in a campaign section
    
    Args:
        soup (soup object): parsed HTML content of a Kickstarter project page
        section (str): label of the campaign section to be analyzed
    
    Returns:
        an integer representing the number of non-YouTube videos in a campaign
        section"""
    
    # Use tree parsing to count all non-YouTube video tags depending on which
    # section is requested
    if section == 'about':
        return len(soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
                '-media formatted-lists'
        ).find_all('div', class_='video-player'))
    elif section == 'risks':
        return len(soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('div', class_='video-player'))

# Display 'NaN' if the section isn't found, otherwise display the # of
# non-YouTube videos
if campaign['about'] == 'section_not_found':
    print(np.nan)
else:
    print(count_videos(soup, 'about'))

def count_youtube(soup, section):    
    """Count the number of YouTube videos in a campaign section
    
    Args:
        soup (soup object): parsed HTML content of a Kickstarter project page
        section (str): label of the campaign section to be analyzed
    
    Returns:
        an integer representing the number of YouTube videos in a campaign
        section"""
    
    # Initialize total number of YouTube videos
    youtube_count = 0

    # Use tree parsing to identify all iframe tags depending on which section
    # is requested
    if section == 'about':
        iframes = soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
            '-media formatted-lists'
        ).find_all('iframe')
    elif section == 'risks':
        iframes = soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('iframe')
    
    # Since YouTube videos are contained only in iframe tags, determine which
    # iframe tags contain YouTube videos and count them
    for iframe in iframes:
        # Catch any iframes that fail to include a YouTube source link
        try:
            if 'youtube' in iframe.get('src'):
                youtube_count += 1
        except TypeError:
            pass
    
    return youtube_count

# Display 'NaN' if the section isn't found, otherwise display the # of
# YouTube videos
if campaign['about'] == 'section_not_found':
    print(np.nan)
else:
    print(count_youtube(soup, 'about'))

def count_gifs(soup, section):    
    """Count the number of GIF images in a campaign section
    
    Args:
        soup (soup object): parsed HTML content of a Kickstarter project page
        section (str): label of the campaign section to be analyzed
    
    Returns:
        an integer representing the number of GIF images in a campaign
        section"""
    
    # Initialize total number of GIFs
    gif_count = 0

    # Use tree parsing to select all image tags depending on the section
    # requested
    if section == 'about':
        images = soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
            '-media formatted-lists'
        ).find_all('img')
    elif section == 'risks':
        images = soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('img')
    
    # Since GIFs are contained in image tags, determine which image tags
    # contain GIFs and count them
    for image in images:
        # Catch any iframes that fail to include an image source link
        try:
            if 'gif' in image.get('data-src'):
                gif_count += 1
        except TypeError:
            pass
    
    return gif_count

# Display 'NaN' if the section isn't found, otherwise display the # of
# GIFs
if campaign['about'] == 'section_not_found':
    print(np.nan)
else:
    print(count_gifs(soup, 'about'))

def count_hyperlinks(soup, section):    
    """Count the number of hyperlink tags in a campaign section
    
    Args:
        soup (soup object): parsed HTML content of a Kickstarter project page
        section (str): label of the campaign section to be analyzed
    
    Returns:
        an integer representing the number of hyperlinks in a campaign
        section"""
    
    # Use tree parsing to compute number of hyperlink tags depending on the
    # section requested
    if section == 'about':
        return len(soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
                '-media formatted-lists'
        ).find_all('a'))
    elif section == 'risks':
        return len(soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('a'))

# Display 'NaN' if the section isn't found, otherwise display the # of
# hyperlinks
if campaign['about'] == 'section_not_found':
    print(np.nan)
else:
    print(count_hyperlinks(soup, 'about'))

def count_bolded(soup, section):    
    """Count the number of bold tags in a campaign section
    
    Args:
        soup (soup object): parsed HTML content of a Kickstarter project page
        section (str): label of the campaign section to be analyzed
    
    Returns:
        an integer representing the number of bolded text tags in a campaign
        section"""
    
    # Use tree parsing to compute number of bolded text tags depending on which
    # section is requested
    if section == 'about':
        return len(soup.find(
            'div',
            class_='full-description js-full-description responsive' + \
                '-media formatted-lists'
        ).find_all('b'))
    elif section == 'risks':
        return len(soup.find(
            'div',
            class_='mb3 mb10-sm mb3 js-risks'
        ).find_all('b'))

# Display 'NaN' if the section isn't found, otherwise display the # of
# bolded text tags and its %
if campaign['about'] == 'section_not_found':
    print(np.nan, np.nan)
else:
    print(
        count_bolded(soup, 'about'),
        count_bolded(soup, 'about') / num_words
    )

# Extract all features for the given section. If the section isn't available, 
# then return 'NaN' for each feature.
section = 'about'
if campaign[section] == 'section_not_found':
    print([np.nan] * 20)
else:
    row = ( 
        len(get_sentences(campaign[section])),
        len(get_words(campaign[section])),
        len(identify_allcaps(campaign[section])),
        len(identify_allcaps(campaign[section])) / num_words,
        count_exclamations(campaign[section]),
        count_exclamations(campaign[section]) / num_words,
        count_apple_words(campaign[section]),
        count_apple_words(campaign[section]) / num_words,
        compute_avg_words(campaign[section]),
        count_paragraphs(soup, section),
        compute_avg_sents_paragraph(soup, section),
        compute_avg_words_paragraph(soup, section),
        count_images(soup, section),
        count_videos(soup, section),
        count_youtube(soup, section),
        count_gifs(soup, section),
        count_hyperlinks(soup, section),
        count_bolded(soup, section),
        count_bolded(soup, section) / num_words,
        campaign[section]
    )
    
    print(pd.Series(row))

