from selenium import webdriver
import re
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
import requests

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx

os.chdir('/home/brian/Documents/aur/html/package-pages/')
driver = webdriver.PhantomJS()
base_url = 'https://www.archlinux.org/packages/?page='
for i in range(1,101):
    search_url = base_url + str(i) + "&"
    print(search_url)
    driver.get(search_url)
    time.sleep(2 + np.random.random())
    html = driver.page_source.encode('utf-8')
    name = "package_page_" + str(i)
    package_list = open(name+'.txt', 'w+')
    package_list.write(str(html))
    package_list.close()

os.chdir('/home/brian/Documents/aur/html/package-pages/')
files = os.listdir()
dict_list = []
for file in files:     
    f = open(file, 'r')
    html = f.read()
    b = BeautifulSoup(html, 'lxml')
    try: 
        packages = b.find_all('tr')[1:]
        for package in packages:
            data = package.find_all('td')
            data_dict = {
                         "arch": data[0].text,
                         "repo": data[1].text,
                         "name": data[2].text,
                         "link": data[2].find('a')['href'],
                         "version": data[3].text,
                         "description": data[4].text, 
                         "last_updated": data[5].text,
                         "flag_date": data[6].text
                        }
            dict_list.append(data_dict)
    except:
        print(file)
    b.decompose()
    f.close()

cols = ['arch', 'repo', 'name', 'link', 'version', 'description', 'last_updated', 'flag_date']
df = pd.DataFrame(dict_list, columns=cols)
df = df.drop_duplicates()
df.to_csv('../csv/arch_pack_data.csv')

os.chdir('/home/brian/Documents/aur/csv/')
df = pd.read_csv('../csv/arch_pack_data.csv', index_col=0)

df.shape

os.chdir('/home/brian/Documents/aur/html/package-details/')
package_html_files = os.listdir()
base_url = "https://www.archlinux.org"
for _, package in df.iterrows():
    file_name = str(package['name'] + '.txt')
    if file_name not in package_html_files: 
        print(f'Getting: {file_name}')
        link = package['link']
        html = requests.get(base_url + link).text
        time.sleep(2)
        f = open(file_name, 'w+')
        f.write(str(html))
        f.close()
        print(f'Finished getting: {file_name}')
    else: 
        print(f'Skip: {_}')

pkg_dict_list = []

err_count = 0
err_files = []
os.chdir('/home/brian/Documents/aur/html/package-details/')
for _, file in enumerate(os.listdir()):
    print(_, end=" ")
    if file != 'ghostdriver.log':
        f = open(file, 'r')
        html = f.read()
        b = BeautifulSoup(html, 'lxml')
        try:
            print(_)
            # attributes
            pkginfo = b.find('table', attrs={'id':'pkginfo'}).find_all('tr')
            
            attr_dict = {}
            for attribute in pkginfo:
                label = attribute.find('th').text.strip(": ")
                value = attribute.find('td').text
                
                if label == "Maintainers":
                    if "Orphan" in value:
                        attr_dict[label] = "Orphan"
                        primary_maintainer = "Orphan"
                        num_maintainers = 0
                    else:
                        values = attribute.find('td').find_all('a')
                        values = [i.text for i in values]
                        num_maintainers = len(values)
                        attr_dict["num_maintainers"] = num_maintainers
                        primary_maintainer = values[0]
                        attr_dict["primary_maintainer"] = primary_maintainer
                        attr_dict[label] = values
                
                elif label == "License(s)":
                    values = value.split(",")
                    license_count = len(values)
                    primary_license = values[0]
                    attr_dict["primary_license"] = primary_license
                    attr_dict[label] = values
                else: 
                    value = value.replace('\\n', '\n')
                    value = value.replace('\\t', '\t')
                    value = re.sub('[\t+]', '', value)
                    value = re.sub('[\n+]', '', value)
                    attr_dict[label] = value
                
            # title and version
            title = b.find('h2').text
            pkg_name = title.split(" ")[0]
            version_number = title.split(" ")[1]
            
            attr_dict['package_name'] = pkg_name
            attr_dict['version_number'] = version_number

            dependencies = []
            pkgdeps = b.find('ul', attrs={'id':'pkgdepslist'})
            if pkgdeps:
                for p in pkgdeps.find_all('li'):
                    for link in p.find_all('a'):
                        dependencies.append(link.text)

            attr_dict['pkgdeps'] = dependencies
            
            requirements = []
            pkgreqs = b.find('ul', attrs={'id':'pkgreqslist'})
            if pkgreqs:
                for p in pkgreqs.find_all('li'):
                    for link in p.find_all('a'):
                        requirements.append(link.text)

            attr_dict['pkgreqs'] = requirements

            pkg_dict_list.append(attr_dict)

        except: 
            err_count += 1
            err_files.append(file)
            print("ERROR")
            print(err_count)
            print(file)
        b.decompose()
        f.close()

df = pd.DataFrame(pkg_dict_list)

df.to_pickle('/home/brian/Documents/aur/pickle/core_packages_df.p')

df = pd.read_pickle('/home/brian/Documents/aur/pickle/core_packages_df.p')

df["requirements_count"] = [len(r) for r in df.pkgreqs]
df["dependencies_count"] = [len(d) for d in df.pkgdeps]

df.requirements_count.sort_values(ascending=False)



