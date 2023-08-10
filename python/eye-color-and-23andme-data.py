import os
import requests
import tempfile
from ohapi import api
print("Checking for 23andMe data in Open Humans...\n")

user = api.exchange_oauth2_member(os.environ.get('OH_ACCESS_TOKEN'))
for entry in user['data']:
    if entry['source'] == "direct-sharing-128" and 'vcf' not in entry['metadata']['tags']:
        file_url_23andme = entry['download_url']
        break
        
if 'file_url_23andme' not in locals():
    print("Sorry, you first need to add 23andMe data to Open Humans!\n"
          "You can do that here: https://www.openhumans.org/activity/23andme-upload/")
else:
    print("Great, you have 23andMe data in Open Humans! We'll retrieve this...\n")

file_23andme = tempfile.NamedTemporaryFile()
file_23andme.write(requests.get(file_url_23andme).content)
file_23andme.flush()

print("Done!")

snps = {
    'rs12913832': None,
    'rs16891982': None,
    'rs12203592': None
}

file_23andme.seek(0)
for line in file_23andme:
    line = line.decode('utf-8').strip()
    if line.startswith('#'):
        continue
    line_data = line.split('\t')
    if line_data[0] in snps.keys():
        snps[line_data[0]] = line_data[3]

for snp in snps.keys():
    print('{}:\t{}'.format(snp, snps[snp] if snps[snp] else 'Unknown'))

your_genotype = ('{}'.format(snps['rs12203592']), '{}'.format(snps['rs12913832']), '{}'.format(snps['rs16891982']))

opensnp_eyecolors = requests.get('https://drive.google.com/uc?export=view&id=1KYeLz0hoSnyv2jHYHiKqLhv2akiIb5xr').json()
opensnp_rs12203592 = requests.get('https://drive.google.com/uc?export=view&id=1opmYjbG_0nVSzw3l0iuFLRVUmZ8LvC80').json()
opensnp_rs12913832 = requests.get('https://drive.google.com/uc?export=view&id=15f9lFEmRsHEFvZskPAzy_l7V3YBDyjeg').json()
opensnp_rs16891982 = requests.get('https://drive.google.com/uc?export=view&id=1yPC4d4hWljODlHWDS9b1M_NTodbslJRl').json()

eyecolor_by_uid = {item['user_id']: item['variation'].lower() for item in opensnp_eyecolors['users']}
rs12203592_by_uid = {item['user']['id']: item['user']['genotypes'][0]['local_genotype'] for item in
                     opensnp_rs12203592 if item['user']['genotypes']}
rs12913832_by_uid = {item['user']['id']: item['user']['genotypes'][0]['local_genotype'] for item in
                     opensnp_rs12913832 if item['user']['genotypes']}
rs16891982_by_uid = {item['user']['id']: item['user']['genotypes'][0]['local_genotype'] for item in
                     opensnp_rs16891982 if item['user']['genotypes']}

joint_uids = [uid for uid in eyecolor_by_uid.keys() if uid in rs12203592_by_uid and
              uid in rs12913832_by_uid and uid in rs16891982_by_uid]

genotypes_to_color = {}
for uid in joint_uids:
    genotype = ('{}'.format(rs12203592_by_uid[uid]),
                '{}'.format(rs12913832_by_uid[uid]),
                '{}'.format(rs16891982_by_uid[uid]))
    if genotype in genotypes_to_color:
        genotypes_to_color[genotype].append(eyecolor_by_uid[uid])
    else:
        genotypes_to_color[genotype] = [eyecolor_by_uid[uid]]

color_counts = {}
for color in genotypes_to_color[your_genotype]:
    if color in color_counts:
        color_counts[color] += 1
    else:
        color_counts[color] = 1

color_counts = sorted(list(color_counts.items()), key=lambda x: x[1], reverse=True)
color_count_sum = sum([item[1] for item in color_counts])
color_count_percentages = [(item[0], item[1]/color_count_sum) for item in color_counts]

print("\nOut of {} people sharing this genotype in openSNP data, they report...\n".format(color_count_sum))
for item in color_count_percentages:
    print('{0:.0f}%\t{1}'.format(item[1]*100, item[0]))

