# import necessary libraries
import os, csv
# other options:   import os.system(wget http)   OR   import urllib.urlretrieve (this is an alternative command)

sample = [] # make empty list
with open('/hdir/0/jhaber/Dropbox/projects/Workshops_and_coding/charter_project/charters_microsam.csv', 'r', encoding = 'Windows-1252')as csvfile: # open file; the windows-1252 encoding looks weird but works for this
    reader = csv.DictReader(csvfile) # create a reader
    for row in reader: # loop through rows
        sample.append(row) # append each row to the list

# Take a look at the contents of our list called sample--just the first entry
print(sample[1]["SEARCH: NAME ADDRESS"], "\n", sample[1]["URL"], "\n", sample[1])

# turning this into tuples we can use with wget!
# first, make some empty lists
url_list = []
name_list = []
terms_list = []

# now let's fill these lists with content from the sample
for school in sample:
    url_list.append(school["URL"])
    name_list.append(school["SCHNAM"])
    terms_list.append(school["SEARCH: NAME ADDRESS"])

# it's VERY important that these three lists be indexed the same, so let's check:
print(url_list[:3], "\n", name_list[:3], "\n", terms_list[:3])

tuple_list = list(zip(url_list, name_list))
# Let's check what these tuples look like:
print(tuple_list[:3])
print("\n", tuple_list[1][1].title())

os.chdir('/hdir/0/jhaber/Documents/projects/charter_data/wget_sept8')
os.getcwd()

k=0 # initialize this numerical variable k, which keeps track of which entry in the sample we are on.
for tup in tuple_list:
    k += 1 # Add one to k, so we start with 1 and increase by 1 all the way up to entry # 300
    print("Capturing website data for", (tup[1].title()) + ", which is school #" + str(k), "of 300...")
    # use the tuple to create a name for the folder
    if k < 10: # Add two zeros to the folder name if k is less than 10 (for ease of organizing the output folders)
        dirname = "00" + str(k) + " " + (tup[1].title())
    elif k < 100: # Add one zero if k is less than 100
        dirname = "0" + str(k) + " " + (tup[1].title())
    else: # Add nothing if k>100
        dirname = str(k) + " " + (tup[1].title())
    os.chdir('/hdir/0/jhaber/Documents/projects/charter_data/wget_sept8')  # cd into data directory--this is a/
    # key step because otherwise wget puts the output into whatever folder it was previously in, can get real messy.
    os.makedirs(dirname) # create the folder using the dirname we just made, then change into that directory
    os.chdir(dirname)  # other options to think about for wget: --page-requisites --retry-connrefused --convert-links --wait=3
    os.system('wget --no-parent --show-progress --progress=dot --recursive --level=3 --convert-links --retry-connrefused         --random-wait --no-cookies --secure-protocol=auto --no-check-certificate --execute robots=off         --user-agent="Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:11.0) Gecko/20100101 Firefox/11.0"         --reject .mov,.MOV,.avi,.AVI,.mpg,.MPG,.mpeg,.MPEG,.mp3,.MP3,.mp4,.MP4,.png,.PNG,.gif,.GIF,.jpg,.JPG,        .jpeg,.JPEG,.pdf,.PDF,.pdf_,.PDF_,.doc,.DOC,.docx,.DOCX,.xls,.XLS,.xlsx,.XLSX,.csv,.CSV,.ppt,.PPT,.pptx,.PPTX'              + ' ' + (tup[0]))

# Grabbing #142--a school that has no directory hierarchy in its website:
os.chdir('/Users/Jaren/Dropbox/projects/Workshops_and_coding/charter_project/urls_extracted/142 Chinook Montessori Charter School')

for url in ['http://www.k12northstar.org/chinook', 'http://www.k12northstar.org/site/Default.aspx?PageID=2678', 'http://www.k12northstar.org/Page/2684', 'http://www.k12northstar.org/Page/2685', 'http://www.k12northstar.org/Page/2686', 'http://www.k12northstar.org/Page/2683', 'http://www.k12northstar.org/Page/2704']:
    os.system('wget --no-parent --show-progress --progress=dot --convert-links         --no-cookies --secure-protocol=auto --no-check-certificate --execute robots=off         --user-agent="Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:11.0) Gecko/20100101 Firefox/11.0"         --reject .mov,.mpg,.mpeg,.mp3,.png,.gif,.jpg,.jpeg,.pdf,.pdf_,.doc,.xls,.xlsx,.csv,.ppt,.pptx' + ' ' + url)

