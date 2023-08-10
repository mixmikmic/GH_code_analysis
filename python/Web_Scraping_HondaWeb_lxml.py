import requests, lxml.html
from getpass import getpass

s = requests.session()

login_url = 'https://hondasites.com/auth/default.html'

login = s.get(login_url)
login_html = lxml.html.fromstring(login.text)
hidden_inputs = login_html.xpath(r'//form//input[@type="hidden"]')

# Create Python dictionary containing key-value pairs of hidden inputs
form = {x.attrib["name"]: x.attrib["value"] for x in hidden_inputs}
print(form)

s = requests.session()
login_url = 'https://hondasites.com/auth/default.aspx'
login_url2 = 'https://myhondda.hondasites.com/_layouts/15/Authenticate.aspx?Source=/'
login_url3 = 'https://myhondda.hondasites.com/_layouts/accessmanagersignin.aspx?ReturnUrl=/_layouts/15/Authenticate.aspx?Source=%2F&Source=/'
login_url4 = 'https://myhondda.hondasites.com/_layouts/15/Authenticate.aspx?Source=/'
login_url5 = 'https://myhondda.hondasites.com/Person.aspx?accountname=i:0%23.f|AccessManagerMembershipProvider|17151'

username = getpass('User Name:')
password = getpass('Password:')

credentials = {
    'username': username,
    'password': password,
    'login_referrer': '',
    'login': 'Y'
}

request1 = s.post(login_url, data=credentials)
print('request1:', request1.status_code)
request2 = s.get(login_url2)
print('request2:', request2.status_code)
request3 = s.get(login_url3)
print('request3:', request3.status_code)
request4 = s.get(login_url4)
print('request4:', request4.status_code)
request5 = s.get(login_url5)
print('request5:', request5.status_code)

request5.content[:500]

profile_html = lxml.html.fromstring(request5.content)
# Get div tag with id="ct100_blah_blah" and span tag with class="ms-tableCell ms-profile-detailsValue", then text()
skills_div = profile_html.xpath(r'//div[@id="ctl00_SPWebPartManager1_g_402dacf0_24c9_49f7_b128_9a852fc0ae8a_ProfileViewer_SPS-Skills"]/span[@class="ms-tableCell ms-profile-detailsValue"]/text()')

if skills_div:
    print('User Skills:', skills_div[0])
else:
    print('User did not enter skills.')

base_profile_url = 'https://myhondda.hondasites.com/Person.aspx?accountname=i:0%23.f|AccessManagerMembershipProvider|'

members = ['17151', '38623', '10770']
for member in members:
    member_url = base_profile_url + member
    request = s.get(member_url)
    profile_html = lxml.html.fromstring(request.content)
    skills_div = profile_html.xpath(r'//div[@id="blah_blah_ProfileViewer_SPS-Skills"]/span[@class="ms-tableCell ms-profile-detailsValue"]/text()')
    if skills_div:
        print('User(', member, ') Skills:', skills_div[0])
    else:
        print('User(', member, ') did not enter skills.')

import requests
import lxml.html
from getpass import getpass

s = requests.session()

login_url = 'https://hondasites.com/auth/default.html'
login_url2 = 'https://myhondda.hondasites.com/_layouts/15/Authenticate.aspx?Source=/'
login_url3 = 'https://myhondda.hondasites.com/_layouts/accessmanagersignin.aspx?ReturnUrl=/_layouts/15/Authenticate.aspx?Source=%2F&Source=/'
login_url4 = 'https://myhondda.hondasites.com/_layouts/15/Authenticate.aspx?Source=/'

base_profile_url = 'https://myhondda.hondasites.com/Person.aspx?accountname=i:0%23.f|AccessManagerMembershipProvider|'

username = getpass('User Name:')
password = getpass('Password:')

credentials = {
    'username': username,
    'password': password,
    'login_referrer': '',
    'login': 'Y'
}

request1 = s.post(login_url, data=credentials)
print('Submitted login')
request2 = s.get(login_url2)
print('Passed authentication #1')
request3 = s.get(login_url3)
print('Passed authentication #2')
request4 = s.get(login_url4)
print('Passed authentication #3')

members = ['17151', '38623', '10770']
for member in members:
    member_url = base_profile_url + member
    request = s.get(member_url)
    profile_html = lxml.html.fromstring(request.content)
    skills_div = profile_html.xpath(r'//div[@id="blah_blah_ProfileViewer_SPS-Skills"]/span[@class="ms-tableCell ms-profile-detailsValue"]/text()')
    if skills_div:
        print('User(', member, ') Skills:', skills_div[0])
    else:
        print('User(', member, ') did not enter skills.')

