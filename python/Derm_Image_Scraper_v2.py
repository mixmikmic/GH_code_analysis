from bs4 import BeautifulSoup
import urllib2
import urlparse 
import re
import os.path
import cPickle as pickle

#Constants - path definitions
HOME_PAGE = "http://www.dermnet.com/dermatology-pictures-skin-disease-pictures/"
DOMAIN_PAGE = "http://www.dermnet.com/"
IMAGE_DIR = "/Users/aum/Documents/Analytics/derm_project2/data"

def openPg(url):
    """Opens a web page for parsing.
    
    Args:
        url: a web address.
    
    Returns:
        BeautifulSoup object to parse.
    """
    html = urllib2.urlopen(url)
    soup = BeautifulSoup(html, "lxml")
    return soup

def getClassImages(class_url):
    """Returns all images for a class label, including all child (webpage) categories.
    
    Args:
        class_url: a web address for a skin disease class.
    
    Returns:
        class_images: A list containing all image links for a class label.
    """
    class_images = []
    cat_urls = getClassCategories(class_url)
    for url in cat_urls:
        class_images.extend(getCategoryImages(url))
    return class_images

def getClassCategories(class_url):
    """Returns all category urls for a skin disease class.
    
    Args:
        class_url: a web address for a skin disease class.
    
    Returns:
        categories: A list containing category urls.
    """
    soup = openPg(class_url)
    cat_links = soup.find("table").find_all("a")
    categories = []
    for link in cat_links:
        abs_link = urlparse.urljoin(DOMAIN_PAGE, link.get('href'))
        categories.append(abs_link)
    return categories

def getCategoryImages(cat_url):
    """Captures all category image urls within a series of paginated links.
    
    Args:
        cat_url: a category web address.
    
    Returns:
        cat_images: A list containing image urls.
    """
    cat_images = []
    cat_thumbpgs = []
    #add to category image list
    addPgImages(cat_url, cat_images)
    cat_thumbpgs = getAllThumbPgs(cat_url)    
    # more pages in category, add images from those thumbnail pages
    if cat_thumbpgs: 
        for page in cat_thumbpgs:
            addPgImages(page, cat_images)
    return cat_images

def getAllThumbPgs(cat_url):
    """Returns pagnated links associated to a category, if any.
    
    Args:
        cat_url: a category web address.
    
    Returns:
        thumb_pgs: A list of pagnated link addresses.
    """
    soup = openPg(cat_url)
    pages = soup.find("div","pagination")
    thumb_pgs = []
    if pages:  #there are multiple pages for this category
        for page in pages:
            if page.name == 'a' and page.string <> 'Next':
                thumb_pgs.append(urlparse.urljoin(DOMAIN_PAGE, page['href']))
    return thumb_pgs

def addPgImages(url,image_list):
    """Finds all image links in a webpage and adds them to the image list.
    
    Args:
        url: a web address for a pagnated category page.
        image_list: a list of image urls
    
    Returns:
        Nothing.
    """
    soup = openPg(url)
    thumbnails = soup.find_all("div","thumbnails")
    if thumbnails: ## there are thumbnails actually on the page
        for thumb in thumbnails:
            thumb_link = thumb.img['src']
            #use full image link instead of thumbnail link
            image_link = re.sub(r'Thumb',"",thumb_link)
            image_list.append(image_link)
            

def createImageDict(dict_file):
    """Create image dictionary and serialize to disk (pickle). Unpickle to dictionary object if already exists.
    
    Args:
        dict_file: Absolute path + filename of pickle file object.
    
    Returns:
        image_dict: dictionary containing image urls for 23 skin disease classes.
    """
    # load dictionary object hierarchy if pickled file exists
    if os.path.exists(dict_file):
        print "Loading image dictionary %s" % dict_file
        with open(dict_file, 'rb') as f:
            try:
                img_dict = pickle.load(f)
                print "Loaded image dictionary."
                return img_dict
            except:
                print "Failure to load: %s. Creating dictionary. " % dict_file
    
    #create dictionary by parsing Dermnet
    #open website root directory and get class links
    soup = openPg(HOME_PAGE)
    class_links = soup.find("table").find_all("a")

    print "Populating image dictionary..."
    img_dict = {}
    for link in class_links:
        abs_link = urlparse.urljoin(DOMAIN_PAGE, link.get('href'))
        class_name = re.sub(r'[^a-z0-9A-Z\s]+', '', link.string)
        #add to final dictionary {class_name: list of image links}
        img_dict[class_name] = getClassImages(abs_link)

    print "Image dictionary populated. Total classes: %s" % len(img_dict)

    #save dictionary to pickle file
    with open(dict_file, 'wb') as f:
        try:
            pickle.dump(img_dict, f)
            print "Saved image dictionary to %s" % dict_file
        except:
            print "Failure to save dictionary %s. \nPlease investigate. " % dict_file

    return img_dict
    


## load existing class-to-image url dictionary, or scrape website.
img_dictionary = createImageDict(os.path.join(IMAGE_DIR,'imageUrls.p'))

## Downloading pictures from dictionary
for key, class_imgs in img_dictionary.iteritems():

    print "Processing class: %s" %key

    #create class folders, if it doesn't exist
    class_path = os.path.join(IMAGE_DIR,key)
    if not os.path.exists(class_path):
        print "Creating dir in: %s" %class_path
        os.mkdir(class_path)

    #check if more images to be added to class dir
    num_dirImgs = len([name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name))])
    count = 0
    if not num_dirImgs == len(class_imgs):
        for img in class_imgs:

            img_name = os.path.basename(img)
            file_name = os.path.join(class_path,img_name)

            if os.path.isfile(file_name):
                print "Skipping: " + img_name + " has already downloaded."
            else:
                #download image
                if class_imgs.index(img) % 10 == 0:
                    print "Downloading image #%s: %s" %(class_imgs.index(img),img_name)
                f = urllib2.urlopen(img).read()
                open(file_name, 'wb').write(f)

    print "Download complete for: %s" %key

print "Scraping Complete."

