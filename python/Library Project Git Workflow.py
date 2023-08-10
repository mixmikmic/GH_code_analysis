get_ipython().system('git status')

get_ipython().system('git add Library\\ Project\\ Git\\ Workflow.ipynb')

get_ipython().system('git status')

get_ipython().system("git commit -m 'Commit example for workflow'")

#Creates a branch called khanacademy
get_ipython().system('git branch khanacademy')

#Checks out (switches to) the khanacademy branch
get_ipython().system('git checkout khanacademy')

get_ipython().system('git add data/khan.py')
get_ipython().system('git status')

get_ipython().system('python data/khan.py')

get_ipython().system("git commit -m 'Got em, coach.'")

get_ipython().system('git checkout master')
get_ipython().system('git merge khanacademy')

# This won't work for you since you can't write to my repo.
# However if you fork my repo on github.com, you'll be able to do this
# by substituting denricoNBHS with your github username
get_ipython().system('git remote add origin https://github.com/denricoNBHS/stem-projects')

get_ipython().system('git push origin master')

