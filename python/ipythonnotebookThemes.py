import itertools
import os
import subprocess
import urllib.request

theme_names = ['3024-dark',
               '3024-light',
               'atelierdune-dark',
               'atelierdune-light',
               'atelierforest-dark',
               'atelierforest-light',
               'atelierheath-dark',
               'atelierheath-light',
               'atelierlakeside-dark',
               'atelierlakeside-light',
               'atelierseaside-dark',
               'atelierseaside-light',
               'bespin-dark',
               'bespin-light',
               'chalk-dark',
               'chalk-light',
               'default-dark',
               'default-light',
               'eighties-dark',
               'eighties-light',
               'grayscale-dark',
               'grayscale-light',
               'greenscreen-dark',
               'greenscreen-light',
               'isotope-dark',
               'isotope-light',
               'londontube-dark',
               'londontube-light',
               'marrakesh-dark',
               'marrakesh-light',
               'mocha-dark',
               'mocha-light',
               'monokai-dark',
               'monokai-light',
               'ocean-dark',
               'ocean-light',
               'paraiso-dark',
               'paraiso-light',
               'railscasts-dark',
               'railscasts-light',
               'shapeshifter-dark',
               'shapeshifter-light',
               'solarized-dark',
               'solarized-light',
               'tomorrow-dark',
               'tomorrow-light',
               'twilight-dark',
               'twilight-light']

theme_names = [
               'greenscreen-dark',
               'greenscreen-light',
               'monokai-dark',
               'monokai-light',
               'solarized-dark',
               'solarized-light']

for i in theme_names:
    get_ipython().system('ipython profile create $i')
    profile_dir = get_ipython().getoutput('ipython locate profile $i')
    url = "https://raw.githubusercontent.com/nsonnad/base16-ipython-notebook/master/base16-" + i + ".css"
    tgt = os.path.join(profile_dir[0], 'static', 'custom', "custom.css")
    print(tgt)
    urllib.request.urlretrieve (url, tgt)



