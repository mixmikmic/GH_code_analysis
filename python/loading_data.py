# Create a policies instance
from privacy_bot.analysis.policies_snapshot_api import Policies
from collections import Counter

policies = Policies()

# policy.html            ---    html format
# policy.text            ---    text format
# policy.domain          ---    domain
# policy.lang            ---    lang
# policy.tld             ---    top level domain

c = Counter()

# Iterate on all policies
for policy in policies:
    # count policies by language
    c[policy.lang] += 1

print(c)

# Similarly we can access some meta info on policies in the data

print("DOMAINS: ", policies.domains)
print("-------------------------")
print("TLDs: ", policies.tlds)
print("-------------------------")
print("LANGUAGES: ", policies.languages)

# Counter of top level domains
tlds = Counter()

#loading only policies in english
for policy in policies.query(lang='en', domain=None, tld=None):
    tlds[policy.tld] += 1

print(tlds)

# We can retrieve available policies of a given domain or company
list(policy.domain for policy in policies.query(domain='google'))

# Accessing a particular policy
google = next(policies.query(domain='google.de'))



def fix_encoding(content):
    return content.encode('latin-1').decode('utf-8')


# first 300 characters
fix_encoding(google.text[:300])





