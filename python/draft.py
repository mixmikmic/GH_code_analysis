import ast
from bs4 import BeautifulSoup
import json
import pandas as pd
import requests
import random
import urlparse

def requests_to_vt(ip):
    vt_host = "https://www.virustotal.com"
    agent_list = ["Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2228.0 Safari/537.36",
                  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2227.0 Safari/537.36",
                  "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2227.0 Safari/537.36",
                  "Mozilla/5.0 (Windows NT 6.4; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2225.0 Safari/537.36",
                  "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2225.0 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2062.124 Safari/537.36",
                  "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2049.0 Safari/537.36",
                  "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.1985.67 Safari/537.36",
                  "Mozilla/5.0 (X11; OpenBSD i386) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.1985.125 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.1916.47 Safari/537.36",
                  "Mozilla/5.0 (Windows NT 6.3; rv:36.0) Gecko/20100101 Firefox/51.0",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10; rv:33.0) Gecko/20100101 Firefox/51.0",
                  "Mozilla/5.0 (X11; Linux i586; rv:31.0) Gecko/20100101 Firefox/51.0",
                  "Mozilla/5.0 (Windows NT 6.1; rv:27.3) Gecko/20130101 Firefox/51.3",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                 ]

    User_Agent = random.choice(agent_list)

    headers = {"Origin" : "https://www.virustotal.com",
               "Accept-Encoding" : "gzip, deflate",
               "User-Agent" : User_Agent,
               "Referer" : "https://www.virustotal.com/ko/",
               "DNT" : "1",
               "cache-control" : "max-age=0",
               "upgrade-insecure-requests" : "1",
               "Cookie" : "VT_CSRF=a61efaaa2ba1749ec1d7dd7a35d7bc0d; VT_PREFERRED_LANGUAGE=en"}
    response = requests.get("https://www.virustotal.com/ko/ip-address/%s/information/" %str(ip), headers=headers)
    data = response.text
    soup = BeautifulSoup(data)
    return soup

soup = requests_to_vt("8.8.8.8")

def vt_to_country(soup):
# <div class="enum-container">
#     <div class="enum">
#       <div class="floated-field-key">Country</div>
#       <div class="floated-field-value"><img class="flag flag-us"> US</div>
#       <br style="clear:both;"/>
#     </div>
#     <div class="enum">
#       <div class="floated-field-key">Autonomous System</div>
#       <div class="floated-field-value">15169 (Google Inc.)</div>
#       <br style="clear:both;"/>
#     </div>
#   </div>

    contents = soup.findAll("div", { "class" : "enum-container"})
    contents = contents[0]
    contents = contents.findAll("div", { "class" : "enum" })
    
    result = ""
    for content in contents:
        result += content.contents[3].text.strip()
        result += " "
    return result.strip()
country = vt_to_country(soup)

def vt_to_dns(soup):
    # raw html
    # <div id="dns-resolutions" style="word-wrap:break-word;">
    #   <div class="enum">
    #     2016-11-22
    #     <a class="margin-left-1" target="_blank" href="/ko/domain/l45wji.35973.hk/information/">l45wji.35973.hk</a>
    #   </div>

    description = "VirusTotal's passive DNS only stores address records. The following domains resolved to the given IP address."
    contents = soup.findAll("div", { "id" : "dns-resolutions"})
    contents = contents[0]
    contents = contents.findAll("div", { "class" : "enum" })

#     print "Description : {0}".format(description)

    result = []
    for content in contents:
        t = content.contents[0].strip()
        domain = content.a.text.strip()
        result.append({"time" : t, "domain" : domain})
#         print "{0}\t{1}".format(t, domain)
    return result

dns = vt_to_dns(soup)

dns

def vt_to_detected_urls(soup):
    # raw html
    # <div id="detected-urls" style="word-wrap:break-word;">
    #   <div class="enum ">
    #     <span class="text-red vt-width-5">1/68</span>
    #     <span>2016-11-20 12:56:27</span>
    #     <a class="margin-left-1" target="_blank" href="/ko/url/1fb0effdf7d8a5df17b782f62fe108a984be5ba89123069a7de599381b613127/analysis/">
    #       http://rogers1.pw/
    #     </a>
    #   </div>
    vt_host = "https://www.virustotal.com"
    description = "Latest URLs hosted in this IP address detected by at least one URL scanner or malicious URL dataset"
    contents = soup.findAll("div", { "id" : "detected-urls"})
    contents = contents[0]
    contents = contents.findAll("div", { "class" : "enum" })

#     print "Description : {0}".format(description)
    
    result = []
    for content in contents:
        ratio = content.span.text
        url = content.a.text.strip()
        t = content.contents[3].text

        # don't print vt_url
        vt_url = content.find("a")['href']
        vt_url = urlparse.urljoin(vt_host, vt_url)
        result.append({"time" : t, "ratio" : ratio, "vt_url" : vt_url, "url" : url})
#         print "{0}\t{1}\t{2}".format(t, ratio, url)
    return result

detected_urls = vt_to_detected_urls(soup)

detected_urls

def vt_to_detected_downloaded(soup):
    # raw html
    # <div id="detected-downloaded"> 
    #   <div class="enum ">
    #     <span class="text-red vt-width-5">1/46</span>
    #     <span>2013-05-30 16:45:57</span>
    #     <a class="margin-left-1" target="_blank" href="/ko/file/d29a629317d7b608e748036c02b952545d595262ede691569eda17a405ed94f7/analysis/">
    #       d29a629317d7b608e748036c02b952545d595262ede691569eda17a405ed94f7
    #     </a>
    #   </div>
    vt_host = "https://www.virustotal.com"
    description = "Latest files that are detected by at least one antivirus solution and were downloaded by VirusTotal from the IP address provided"
    contents = soup.findAll("div", { "id" : "detected-downloaded"})
    contents = contents[0]
    contents = contents.findAll("div", { "class" : "enum" })

    result = []
    for content in contents:
        t = content.contents[3].text
        ratio = content.span.text
        vt_url = content.a['href']
        url = urlparse.urljoin(vt_host, vt_url)
        result.append({"time" : t, "ratio" : ratio, "vt_url" : vt_url, "url" : url})

#         print "{0}\t{1}\t{2}".format(t, ratio, url)
    return result

detected_downloaded = vt_to_detected_downloaded(soup)

detected_downloaded

def vt_to_detected_communicating(soup):
    # raw html
    # <div id="detected-communicating">  
    #   <div class="enum ">
    #     <span class="text-red vt-width-5">45/57</span>
    #     <span>2016-10-22 12:15:37</span>
    #     <a class="margin-left-1" target="_blank" href="/ko/file/7f2adb4a7e6b2ead8d6adb18f11b81e9860489cd59ed7a370169c751ab00ea64/analysis/">
    #       7f2adb4a7e6b2ead8d6adb18f11b81e9860489cd59ed7a370169c751ab00ea64
    #     </a>
    #   </div>
    vt_host = "https://www.virustotal.com"
    description = "Latest files submitted to VirusTotal that are detected by one or more antivirus solutions and communicate with the IP address provided when executed in a sandboxed environment."
    contents = soup.findAll("div", { "id" : "detected-communicating"})
    contents = contents[0]
    contents = contents.findAll("div", { "class" : "enum" })

#     print "Description : {0}".format(description)
    result = []
    for content in contents:
        t = content.contents[3].text
        ratio = content.span.text
        vt_url = content.a['href']
        url = urlparse.urljoin(vt_host, vt_url)
        result.append({"time" : t, "ratio" : ratio, "vt_url" : vt_url, "url" : url})
#         print "{0}\t{1}\t{2}".format(t, ratio, url)
    return result

detected_communicating = vt_to_detected_communicating(soup)

detected_communicating

def vt_to_detected_referrer(soup):
    # raw html
    #  <div id="detected-referrer">
    #   <div class="enum ">
    #     <span class="text-red vt-width-5">23/53</span>
    #     <a class="margin-left-1" target="_blank" href="/ko/file/e3cbba5d6418e4324e912f937a4789a4c20dd5708b971f886ce4258958a1bfd5/analysis/">
    #       e3cbba5d6418e4324e912f937a4789a4c20dd5708b971f886ce4258958a1bfd5
    #     </a>
    #   </div>
    vt_host = "https://www.virustotal.com"
    description = "Latest files that are detected by at least one antivirus solution and embed URL pattern strings with the IP address provided"
    contents = soup.findAll("div", { "id" : "detected-referrer"})
    contents = contents[0]
    contents = contents.findAll("div", { "class" : "enum" })

#     print "Description : {0}".format(description)
    result = []
    for content in contents:
        ratio = content.span.text
        path = content.a['href']
        vt_url = urlparse.urljoin(vt_host, path)
        result.append({"ratio" : ratio, "vt_url" : vt_url})
#         print "{0}\t{1}".format(ratio, url)
        
        return result

detected_referrer = vt_to_detected_referrer(soup)

detected_referrer

# get info
try:
    country = vt_to_country(soup)
except Exception as e:
    print "[+] Fail to parse country"
    print e
try:
    dns = vt_to_dns(soup)
except Exception as e:
    print "[+] Fail to parse dns"
    print e
    
try:
    detected_urls = vt_to_detected_urls(soup)
except Exception as e:
    print "[+] Fail to parse detected_urls"
    print e

try:
    detected_downloaded = vt_to_detected_downloaded(soup)
except Exception as e:
    print "[+] Fail to parse detected_downloaded"
    print e
    
try:
    detected_communicating = vt_to_detected_communicating(soup)
except Exception as e:
    print "[+] Fail to parse detected_communicating"
    print e
    
try:
    detected_referrer = vt_to_detected_referrer(soup)
except Exception as e:
    print "[+] Fail to detected_referrer"
    print e

def check_ratio(items, ratio=10):
    result = []
    if len(items) > 0:
        for item in items:
            if int(item['ratio'].split("/")[0]) > ratio:
                result.append(item)
                print item
    return result

# more than 10 av solutions 
detected_urls = check_ratio(detected_urls, 10)
detected_downloaded = check_ratio(detected_downloaded, 10)
detected_communicating = check_ratio(detected_communicating, 10)
detected_referrer = check_ratio(detected_referrer, 10)

# Summary
print "[+] VirusTotal Result Summary"
print "[+] Country is {0}".format(country)
print "[+] The number of domain is {0}".format(len(dns))
print "[+] The number of detected_urls is {0}".format(len(detected_urls))
print "[+] The number of detected_downloaded is {0}".format(len(detected_downloaded))
print "[+] The number of detected_communicating is {0}".format(len(detected_communicating))
print "[+] The number of detected_referrer is {0}".format(len(detected_referrer))

def request_safebrowsing(ip):
    agent_list = ["Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2228.0 Safari/537.36",
                  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2227.0 Safari/537.36",
                  "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2227.0 Safari/537.36",
                  "Mozilla/5.0 (Windows NT 6.4; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2225.0 Safari/537.36",
                  "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2225.0 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2062.124 Safari/537.36",
                  "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2049.0 Safari/537.36",
                  "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.1985.67 Safari/537.36",
                  "Mozilla/5.0 (X11; OpenBSD i386) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.1985.125 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.1916.47 Safari/537.36",
                  "Mozilla/5.0 (Windows NT 6.3; rv:36.0) Gecko/20100101 Firefox/51.0",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10; rv:33.0) Gecko/20100101 Firefox/51.0",
                  "Mozilla/5.0 (X11; Linux i586; rv:31.0) Gecko/20100101 Firefox/51.0",
                  "Mozilla/5.0 (Windows NT 6.1; rv:27.3) Gecko/20130101 Firefox/51.3",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2840.98 Safari/537.36",
                 ]

    User_Agent = random.choice(agent_list)

    headers = {"Accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
               "User-Agent" : User_Agent,
               "DNT" : "1",
               "Cookie" : "PREF=ID=7c4801ccf11172b5:U=cf79dbff45e75c9b:FF=0:LD=ko:TM=1415531226:LM=1421140541:S=stJPu1zeIuoqj4eY; NID=67=S6lT0yIpAKOWo8VGcgrs7R8KkqVdaC3-bHTgxAgeTojCthyMbyu1htksbRDSqCrmht5UYQVb7b0WmvHcoj1ks-FStKihS1QOv47HTWx3DIpcWqnTiMEEX7Tm_xt-nnec7BRuqDQ6q0_KfHIIK04T_Sc3IImCKON8Z75HRlESpFY; SID=DQAAAOgAAABpCDvuqMJLeE1PaiXp0oMuiJn2sPmhhpatrUEax7wMbZmF28ei_rgFbTstNePHvnOmXaFC3zshLc-yWvLLLnpXrpE8JjLCDV83KOGNcoEdxIGQo7fzrYvYo6g0EpL7HpIRBmcfOxV86jTNoUUJ7I4uNGuSe3U1V1EorVFuSUEeb5QXxnIgjWZPJgLC5BLVKwFnQYKo_XeRyhseViIY7PmH_IvA9elJ3tnaStpia9jvr9lBYMVH31iPa1VYIL9Tu6fRpCZgy7qMqXczILT6Rm2UR4LucY00UIh879RQOy80BG3h2uLTZ9mTcVCGNDVDwac; HSID=A0oMFQ14zp7IdG69H; SSID=AVBcevuZ2xNm1FZ3T; APISID=t51XxytEHH_0a6_E/AzA8mwzdqqGk-hRYU; SAPISID=OVStu50FtudKGJGH/A879KTQhgtMkXkw1_"}

    response = requests.get("https://www.google.com/safebrowsing/diagnostic?output=jsonp&site=%s" %str(ip), headers=headers)
    response = response.content[response.content.find("{") : response.content.rfind("}")+1]
    response = ast.literal_eval(response)
    return response

safebrowsing = request_safebrowsing("http://politlco.com/")

safebrowsing

safebrowsing = request_safebrowsing("http://amzipalq.com/")

safebrowsing

# Summary
print "[+] Google Safe Browsing Result Summary"
print "[+] URL is {0}".format(safebrowsing['website']['name'])
print "[+] MalwareListStatus is {0}".format(safebrowsing['website']['malwareListStatus'])
print "[+] MalwareDownloadListStatus is {0}".format(safebrowsing['website']['malwareDownloadListStatus'])

pd.DataFrame(detected_urls)

df_detected_urls = pd.DataFrame(detected_urls)
df_detected_urls['description'] = 'detected_urls from virustotal'

df_detected_downloaded = pd.DataFrame(detected_downloaded)
df_detected_downloaded['description'] = 'detected_downloaded from virustotal'

df_detected_communicating = pd.DataFrame(detected_communicating)
df_detected_communicating['description'] = 'detected_communicating from virustotal'

df_detected_referrer = pd.DataFrame(detected_referrer)
df_detected_referrer['description'] = 'detected_referrer from virustotal'

df_detected_urls

pd.DataFrame(data = {'description':['safebrowsing : {0}'.format(safebrowsing)]})

df_safebrowsing = pd.DataFrame(data = {'description':['safebrowsing : {0}'.format(safebrowsing)]})

df = pd.concat([df_detected_urls, df_detected_downloaded, df_detected_communicating, df_detected_referrer, df_safebrowsing])
df['object'] = '8.8.8.8'

df.head(2)

df.to_csv('./{0}.csv'.format("8.8.8.8"), sep=',')
df.to_pickle('./{0}.pickle'.format("8.8.8.8"))

key = json.load(open('./key.json'))

# URL Analysis Request API
params = {'api_key': key['malwares'], 'url':'www.dropbox.com'}
agent_list = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
              "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
              "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36",
              "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1",
             ]
User_Agent = random.choice(agent_list)

headers = {"User-Agent" : User_Agent}
response = requests.post('https://www.malwares.com/api/v2/url/info', headers=headers, data=params)

response.json()

data = response.json()

data.keys()

data['virustotal']

# URL Analysis Request API
def requests_to_malwares_url(url):
    result_code = {
    "2" : "Now analyzing",
    "1" : "Data exists",
    "0" : "Data is not exist",
    "-1" : "Invalid Parameters",
    "-11" : "No matching data to API Key",
    "-12" : "No authority to use",
    "-13" : "Expired API Key",
    "-14" : "Over the daily request limit",
    "-15" : "Over the hourly request limit",
    "-41" : "Invalid type of url",
    "-404" : "No result",
    "-500" : "Internal Server Error"
    }
    params = {'api_key': key['malwares'], 'url':url}
    agent_list = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
                  "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
                  "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36",
                  "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1",
                 ]
    User_Agent = random.choice(agent_list)
    headers = {"User-Agent" : User_Agent}
    try:
        response = requests.post('https://www.malwares.com/api/v2/url/info', headers=headers, data=params, timeout=10)
        
        if (response.status_code == 200) and (response.json()['result_code'] == 1):
            data = response.json()
            url = data['url']
            positives = data['virustotal']['positives']
            smishing = data['smishing']
            return {"result" : result_code[str(data['result_code'])], "url" : url, "positives" : positives, "smishing" : smishing}
        elif response.json()['result_code'] != 0:
            data = response.json()
            return {"result" : result_code[str(data['result_code'])], "url" : url}
    except Exception as e:
        return {"result" : str(e), "url" : url}

print requests_to_malwares_url("kingskillz.ru")

print requests_to_malwares_url("nate.co.kr")

# IP Report API
def requests_to_malwares_ip(ip):
    result_code = {
    "1" : "Data exists",
    "0" : "Data is not exist",
    "-1" : "Invalid Parameters",
    "-11" : "No matching data to API Key",
    "-12" : "No authority to use",
    "-13" : "Expired API Key",
    "-14" : "Over the daily request limit",
    "-15" : "Over the hourly request limit",
    "-51" : "Invalid type of ip",
    "-404" : "No result",
    "-500" : "Internal Server Error"
    }
    agent_list = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
                  "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
                  "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36",
                  "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1",
                 ]
    User_Agent = random.choice(agent_list)

    headers = {"User-Agent" : User_Agent}
    try:
        response = requests.get('https://www.malwares.com/api/v2/ip/info?api_key={0}&ip={1}'.format(key['malwares'], "1.2.3.4"), headers=headers, data=params, timeout=10)
        if (response.status_code == 200) and (response.json()['result_code'] == 1):
            data = response.json()
            
            result = {}
            location_cname = response.json()['location']['cname']
            result['location_cname'] = location_cname
            
            location_city = response.json()['location']['city']
            result['location_city'] = location_city
            
            for k in response.json().keys():
                if k == "detected_url":
                    detected_url = response.json()['detected_url']['total']
                    result['detected_url'] = detected_url
                if k == "detected_downloaded_file":
                    detected_downloaded_file = response.json()['detected_downloaded_file']['total']
                    result['detected_downloaded_file'] = detected_downloaded_file
                if k == "detected_communicating_file":
                    detected_communicating_file = response.json()['detected_communicating_file']['total']
                    result['detected_communicating_file'] = detected_communicating_file    
                result["result"] = result_code[str(data['result_code'])]
                result["ip"] = ip
            return result
        elif response.json()['result_code'] != 0:
            data = response.json()
            return {"result" : result_code[str(data['result_code'])], "ip" : ip}
    except Exception as e:
        return {"result" : str(e), "ip" : ip}

requests_to_malwares_ip("8.8.8.8")

agent_list = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
              "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
              "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36",
              "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1",
             ]
User_Agent = random.choice(agent_list)
headers = {"User-Agent" : User_Agent, "Content-Type" : "application/json"}
data =  {
    "client": {
      "clientId":      "malware scan",
      "clientVersion": "1.5.2"
    },
    "threatInfo": {
      "threatTypes":      ["MALWARE", "SOCIAL_ENGINEERING", "POTENTIALLY_HARMFUL_APPLICATION", "THREAT_TYPE_UNSPECIFIED", "UNWANTED_SOFTWARE"],
      "platformTypes":    ["ANY_PLATFORM"],
      "threatEntryTypes": ["URL"],
      "threatEntries": [
        {"url": "http://kingskillz.ru/"},
        {"url" : "ihaveaproblem.info"}
      ]
    }
  }
params = {"key" : key['Safe Browsing']}
r = requests.post("https://safebrowsing.googleapis.com/v4/threatMatches:find", params=params,
                  headers=headers, json=data)

r.json()

r.status_code

d = {'positives': 0, 'response_code': 1, 'total': 68, 'resource': 'http://www.daum.net'}

pd.DataFrame(d, index=['resource'])

pd.DataFrame(d, index=['resource']).to_json("./test.json")

