get_ipython().magic('matplotlib inline')

import os
import json
import itertools

import distance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import brewer2mpl
from gensim.models import tfidfmodel
from gensim.models import lsimodel
from gensim import corpora
from gensim import matutils
from sklearn import manifold
from bokeh import plotting
from bokeh import models
from bokeh import io

from bob_emploi.lib import read_data

io.output_notebook()

TOOLS="pan,wheel_zoom,box_zoom,reset,hover"

data_path = os.getenv('DATA_PATH')

# see gensim_model_creation.ipynb
dict_path = os.path.join(data_path, 'wiki/frwiki_wordids.txt.bz2')
tfidf_model_path = os.path.join(data_path, 'wiki/frwiki.tfidf_model')
lsi_model_path = os.path.join(data_path, 'wiki/frwiki_lsi')
rome_path = os.path.join(data_path, 'rome/ficheMetierXml')

def extract_from_fiche_dict(fiche_dict):
    return {
         'code_rome': fiche_dict['code_rome'],
         'name': fiche_dict['intitule'],
         'description': fiche_dict['description'][0],
         'work_cond': fiche_dict['work_cond'][0],
         'activities': [activity['code_ogr'] for activity in fiche_dict['activities']],
         'skills': [skill['code_ogr'] for skill in fiche_dict['skills']],
         'work_env': [work_env['code_ogr'] for work_env in fiche_dict['work_env']]
    }

fiche_dicts = [read_data.fiche_extractor(fiche) 
               for fiche in read_data.load_fiches_from_xml(rome_path)]
jobs = [extract_from_fiche_dict(fiche_dict) for fiche_dict in fiche_dicts]

# the dictionary is the mapping from actual words to integer IDs
dictionary = corpora.Dictionary.load_from_text(dict_path)
# this model is used to transform new text into Tf-idf
tfidf_model = tfidfmodel.TfidfModel.load(tfidf_model_path)
# transforms new text from Tf-idf to the 400 dim dense LSI space
# which was created by using the Wikipedia as a training corpus
lsi_model = lsimodel.LsiModel.load(lsi_model_path)

tsne = manifold.TSNE(n_components=2, random_state=0)

def text2lsi(text):
    tokens = text.lower().split()
    bow = dictionary.doc2bow(tokens)
    return lsi_model[tfidf_model[bow]]

def gensim_kernel(job1, job2):
    '''! operates on LSI representation of job, not on job'''
    return matutils.cossim(job1, job2)

def superkernel(job1, job2):
    job_1_stuff = job1['work_env'] + job1['skills']
    job_2_stuff = job2['work_env'] + job2['skills']
    return distance.jaccard(job_1_stuff, job_2_stuff)

def workenv_overlap_kernel(job1, job2):
    return distance.jaccard(job1['work_env'], job2['work_env'])

def skill_overlap_kernel(job1, job2):
    return distance.jaccard(job1['skills'], job2['skills'])

def activity_overlap_kernel(job1, job2):
    return distance.jaccard(job1['activities'], job2['activities'])

def distance_matrix(objects, metric):
    res = [[metric(o1, o2) for o2 in objects] for o1 in objects]
    return np.matrix(res)

def plot_embedding(res, jobs, title):
    rome_categories = list(set([j['code_rome'][0] for j in jobs]))
    bmap = brewer2mpl.get_map('Set3', 'Qualitative', 12)
    colors = bmap.hex_colors + ['#7FDA3C', '#D34641']
    job_colors = [colors[rome_categories.index(j['code_rome'][0])] for j in jobs]    
    

    source = plotting.ColumnDataSource(
        data=dict(
            title= [j['code_rome'] + ': ' + j['name'] for j in jobs]
        )
    )
    
    vis_x = res[:, 0]
    vis_y = res[:, 1]    

    p = plotting.figure(title=title, tools=TOOLS)
    p.circle(vis_x, vis_y, radius=0.5, source=source,
             fill_color=job_colors,
             line_color=None, fill_alpha=0.5)

    hover = p.select(dict(type=models.HoverTool))
    hover.tooltips = [
        ("title", "@title")
    ]

    plotting.show(p)    

skill_overlap_dists = distance_matrix(jobs, skill_overlap_kernel)
skill_tsne_res = tsne.fit_transform(skill_overlap_dists) 
plot_embedding(skill_tsne_res, jobs, "skill sets jaccard distance")

activity_overlap_dists = distance_matrix(jobs, activity_overlap_kernel)
act_tsne_res = tsne.fit_transform(activity_overlap_dists) 
plot_embedding(act_tsne_res, jobs, "activity sets jaccard distance")

workenv_overlap_dists = distance_matrix(jobs, workenv_overlap_kernel)
workenv_tsne_res = tsne.fit_transform(workenv_overlap_dists) 
plot_embedding(workenv_tsne_res, jobs, "workenv sets jaccard distance")

combined_overlap_dists = distance_matrix(jobs, superkernel)
combined_tsne_res = tsne.fit_transform(combined_overlap_dists) 
plot_embedding(combined_tsne_res, jobs, "workenv and skill sets jaccard distance")

job_desc_lsi = [text2lsi(job['description']) for job in jobs]
desc_dists = distance_matrix(job_desc_lsi, gensim_kernel)
job_desc_tsne_res = tsne.fit_transform(desc_dists) 
plot_embedding(job_desc_tsne_res, jobs, "job description vector space distance")

job_desc_workenv_lsi = [text2lsi(job['description'] + ' ' + job['work_cond']) for job in jobs]
desc_dists = distance_matrix(job_desc_workenv_lsi, gensim_kernel)

job_desc_workenv_tsne_res = tsne.fit_transform(desc_dists) 
plot_embedding(job_desc_workenv_tsne_res, jobs, "job and work_env description vector space distance")

