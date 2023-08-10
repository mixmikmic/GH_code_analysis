from pickle import dump, load
import pandas
import requests
import urllib.request
from bs4 import BeautifulSoup
import PyPDF2
api_key = 'b5Uc6UQwdVYBhNV20O11AZFc6s2cMZ8tpYrUc9tV' # has increased rate of 2500 -- 1247 comments at a time

def read_file_get_docid(filepath):
    dump_df = load(open(filepath,'rb'))
    df_with_comments = dump_df[dump_df.numberOfCommentsReceived > 0]
    df_with_comments =df_with_comments.sort(['numberOfCommentsReceived'], ascending=[False])
    doc_id = df_with_comments.documentId
    doc_type = df_with_comments.documentType
    #return [doc_id,set(doc_type)]
    return df_with_comments

# [doc_id_list,types] = read_file_get_docid('data/AD_doc_list') #choose the category dump file
# print(types)
# # document ID with 4 parts represent documents. 3 parts represent dockets 
# doc_ids = [doc_id for doc_id in doc_id_list if len(doc_id.split('-')) == 4]
# len(doc_ids)

def download_file(download_url):
    try:
        response = urllib.request.urlopen(download_url)
        file = open("document.pdf", 'wb')
        file.write(response.read())
        file.close()
    except:
        print("(downloading the pdf exception)error log" +  download_url)
    
def get_attached_comments(comment_id, key=api_key):
    #print(each_id) # fro debugging
    #open the api to get file url
    url = "http://api.data.gov:80/regulations/v3/document.json?api_key="+key+"&documentId="+comment_id
    try:
        response = requests.get(url)
    except:
        print("(api opening of attached comment exception)error log" +  url)
        return ""
    if response.status_code != 200:
        print("status code " +str(response.status_code)+" (get_attached_comments) program will break at this point which is ok because we dont need inconsistent data. Run again ")
    data = response.json()
    att_count = 0
    for i in range(len(data["attachments"])):
        if "fileFormats" in data["attachments"][i]:
            att_count = len(data["attachments"][0]["fileFormats"])
    comment_text =""
    for i in range(att_count):
        if data["attachments"][0]["fileFormats"][i].endswith("pdf"):
            link = data["attachments"][0]["fileFormats"][i] 
            access_link = link+'&api_key='+key
            #download file(pdf) and read pdf (page by page)
            download_file(access_link)
            pdfFileObj = open('document.pdf','rb')     #'rb' for read binary mode
            try:
                pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
                pno = pdfReader.numPages
                for i in range(pno):
                    pageObj = pdfReader.getPage(i)          #'i' is the page number
                    comment_text += pageObj.extractText()
            except:
                print("(pdf exception)cant read "+comment_id ) # prints in case we are not able to read file
            break # execute the whole thing for 1st found pdf
    return comment_text

def get_document_comments_from_api(docketId,key=api_key):
    offset=0
    url = "http://api.data.gov:80/regulations/v3/documents.json?api_key="+key+"&countsOnly=1&dct=PS&dktid="+docketId
    try:
        response = requests.get(url)
    except:
        print("(api opening comment count)error log"+url) # prints in case we are not able to read file
    if response.status_code != 200:
        print("status code"+str(response.status_code) + " (get_document_comments_from_api) program will break at this point which is ok because we dont need inconsistent data. Run again")
    data = response.json()
    total = data['totalNumRecords']
    com_list =[]
    for i in range(0,total,1000):
        url = "http://api.data.gov:80/regulations/v3/documents.json?api_key="+key+"&countsOnly=0&&rpp=1000&po="+str(i)+"&dct=PS&dktid="+docketId
        try:
            response = requests.get(url)
        except:
            print("(api opening actual comments)error log"+url) # prints in case we are not able to read file
        #print("Offset:"+str(i)+" Code:"+str(response.status_code))
        if response.status_code != 200:
            print(response.status_code)
        data = response.json()
        com_list += data['documents']
    com_df = pandas.DataFrame(com_list)
    return com_df

def get_document_content_from_api(docId,key=api_key):
    url = "http://api.data.gov:80/regulations/v3/document.json?api_key="+key+"&documentId="+docId
    try:
        response = requests.get(url)
    except:
         print("(api opening doc exception) error log"+url)   
    if response.status_code != 200:
        print("status code "+str(response.status_code)+" (get_document_content_from_api) program will break at this point which is ok because we dont need inconsistent data. Run again")
    data = response.json()
    
    # Get HTML for document content
    if(len(data['fileFormats']) == 2):    
        link = data['fileFormats'][1] # The second link is the document in HTML format
    else:
        link = data['fileFormats'][0]
    access_link = link+'&api_key='+key
    
    try:
        with urllib.request.urlopen(access_link) as response:
            html = response.read()
    except:
        print("doc file opening exception")
    
    # We are interested in the pre tag of the HTML content
    soup = BeautifulSoup(html, "lxml")
    content = soup.find_all('pre')
    
    # Now we need to construct comment_id from document_id
    docket_id = '-'.join(docId.split('-')[:3])
    comment_df = get_document_comments_from_api(docket_id)
    # get comment text where exists
    comment_list =[]
    if not comment_df.empty:
        if "commentText" in comment_df:
            comment_text =comment_df[comment_df.commentText.notnull()].commentText
            comment_list =comment_text.tolist()
        #get doc id where there is attchment
        c_ids = comment_df[comment_df.attachmentCount>0].documentId
        # get comment for each id in list
        for each_id in c_ids.unique():
            comment_list.append(get_attached_comments(each_id))
    doc_dict = {
        "text":content,
        "comment_list":comment_list
    }
    return doc_dict

# data format is array of dicts
def get_one_doc(name,docid):
    #open file - get present data (empty to begin witj)
    filepath = "data/"+name+"_doc_content"
    inp =open(filepath,'rb')
    doc_collection = load(inp)
    inp.close()
    #get that one doc
    #resp = get_document_content_from_api(docid)
    doc_collection.append(docid)
    #put back in file 
    output = open(filepath, 'wb')
    dump(doc_collection, output, -1)
    output.close()

# run this line once every 1 hr
get_one_doc("Master2",one_dc)

one_dc=get_document_content_from_api('SBA-2010-0001-0001')

one_dc['text']
#len(one_dc['comment_list'])

# will return the number of documents read. replace by your name
dc =load(open("data/Master2_doc_content",'rb'))
len(dc)

dc[4]["text"]

df_test =read_file_get_docid('data/CT_doc_list')
df_test.reset_index(inplace = True)

df_test["numberOfCommentsReceived"][10:65] #60,61,62,63

df_test["documentId"][46:64]

master_list =[]

## Getting all of it together
inp =open("data/Emily_doc_content",'rb')
col = load(inp)
inp.close()

for item in col:
    if item['text'] != []:
        master_list.append(item)

master_list.append(col[8])

len(master_list)

master_list[1]['text']

col[8]['text']

len(col)

output = open("data/Master_doc_content", 'wb')
dump(master_list, output, -1)
output.close()



