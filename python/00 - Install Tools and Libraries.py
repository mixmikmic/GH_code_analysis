get_ipython().run_cell_magic('bash', '', '\npip install tensorflow==1.7\npip install tensorflow-transform\npip install tensorflow-model-analysis\npip install google-cloud-dataflow==2.3\npip install python-snappy==5.0')

get_ipython().run_cell_magic('bash', '', '\njupyter nbextension enable --py widgetsnbextension \njupyter nbextension install --py --symlink tensorflow_model_analysis --user')

get_ipython().run_cell_magic('bash', '', '\njupyter nbextension enable tensorflow_model_analysis --py')

get_ipython().run_cell_magic('bash', '', '\ngit clone https://github.com/PAIR-code/facets\ncd facets')

get_ipython().run_cell_magic('bash', '', '\njupyter nbextension install facets-dist/ --user')

