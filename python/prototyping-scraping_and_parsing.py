# Load required libraries
import requests
from bs4 import BeautifulSoup
import lxml

# Select a Kickstarter project page
hyperlink = 'https://www.kickstarter.com/projects/1799891707/ghost-huntin' +     'g-team-and-equipment?ref=recommended'

# Scrape the project page
scraped_html = requests.get(hyperlink)

# Parse the HTML content using an lxml parser
soup = BeautifulSoup(scraped_html.text, 'lxml')

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
    
    # Collect the "Risks and challenges" section if available, and remove #
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
    
    # Clean up both sections and return them in a dictionary
    return {'about': clean_up(section1), 'risks': clean_up(section2)}

# Display the `About this project` section
campaign = get_campaign(soup)
campaign['about']

# Display the `Risks and challenges` section
campaign['risks']

