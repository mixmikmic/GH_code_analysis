"""
An example to show howto retrieve job names and 
details about individual jobs using Jenkinsapi.
"""

from jenkinsapi import api

jenJob = api.Jenkins('https://builds.apache.org/api/python')
jobName_1 = jenJob.keys()[0]
jobName_2 = jenJob.keys()[1]

### Section 1:
### ==========

print(jobName_1)
print(jobName_2)

help(jenJob.jobs)

help(jenJob.jobs.keys)

### Section 2:
### ==========
"""
Print all the jobs using the method keys() found in the class 
Jobs() that's part of 'jobs.py' in the jenkinsapi module.
"""

allJobs = jenJob.jobs.keys()
print("We will be evaluating build details of these projects hosted at Apache software foundation.")
print(allJobs)

### Section 3:
### ==========
"""
Print all the jobs using the method get_jobs_info(), found in the 
class Jenkins() that's part of 'jenkins.py' in the jenkinsapi module.
"""

for info in jenJob.get_jobs_info():
    print(info[0], info[1])

### Section 4:
### ==========

"""
Print all the jobs using the method get_jobs(), found in the 
class Jenkins() that's part of 'jenkins.py' in the jenkinsapi module.

On the info[1], which is reference to the various methods in the class Job 
that's part of the job.py in the jenkinsapi module, it is possible to obtain
details about the builds that belong to that particular job name.
"""

print(jenJob.get_jobs())

"""
Functions using the yield keyword are called generators.
In Section 4, jenjob.get_jobs() is a generator.
It can be iterated using a for-in loop or .next()
"""
print(jenJob.get_jobs().next())
print(jenJob.get_jobs().next())
print(jenJob.get_jobs().next())

for info in jenJob.get_jobs():
    print(info[0])



