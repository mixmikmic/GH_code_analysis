import os
from os import path
import json

get_ipython().system('rm -r flaskwebapp')

get_ipython().system('mkdir flaskwebapp')
get_ipython().system('mkdir flaskwebapp/nginx')
get_ipython().system('mkdir flaskwebapp/etc')

get_ipython().system('cp resnet_v1_152.ckpt flaskwebapp')
get_ipython().system('cp synset.txt flaskwebapp')
get_ipython().system('cp driver.py flaskwebapp')
get_ipython().system('ls flaskwebapp')

get_ipython().run_cell_magic('writefile', 'flaskwebapp/app.py', 'from flask import Flask, request\nimport time\nimport logging\nimport json\nimport driver\n\napp = Flask(__name__)\npredict_for = driver.get_model_api()\n\n\n@app.route(\'/score\', methods = [\'POST\'])\ndef scoreRRS():\n    """ Endpoint for scoring\n    """\n    if request.headers[\'Content-Type\'] != \'application/json\':\n        return Response(json.dumps({}), status= 415, mimetype =\'application/json\')\n    request_input = request.json[\'input\']\n    predictions = predict_for(request_input)\n    return json.dumps({\'result\': predictions})\n\n\n@app.route("/")\ndef healthy():\n    return "Healthy"\n\n\n@app.route(\'/version\', methods = [\'GET\'])\ndef version_request():\n    return driver.version()\n\n\nif __name__ == "__main__":\n    app.run(host=\'0.0.0.0\') # Ignore, Development server')

get_ipython().run_cell_magic('writefile', 'flaskwebapp/wsgi.py', 'import sys\nfrom app import app as application\n\ndef create():\n    print("Initialising")\n    application.run(host=\'127.0.0.1\', port=5000)')

get_ipython().run_cell_magic('writefile', 'flaskwebapp/requirements.txt', 'pillow\nclick==6.7\nconfigparser==3.5.0\nFlask==0.11.1\ngunicorn==19.6.0\njson-logging-py==0.2\nMarkupSafe==1.0\nolefile==0.44\nrequests==2.12.3')

get_ipython().run_cell_magic('writefile', 'flaskwebapp/nginx/app', 'server {\n    listen 80;\n    server_name _;\n \n    location / {\n    include proxy_params;\n    proxy_pass http://127.0.0.1:5000;\n    proxy_connect_timeout 5000s;\n    proxy_read_timeout 5000s;\n  }\n}')

image_name = "masalvar/tfresnet-gpu"
application_path = 'flaskwebapp'
docker_file_location = path.join(application_path, 'dockerfile')

get_ipython().run_cell_magic('writefile', 'flaskwebapp/gunicorn_logging.conf', '\n[loggers]\nkeys=root, gunicorn.error\n\n[handlers]\nkeys=console\n\n[formatters]\nkeys=json\n\n[logger_root]\nlevel=INFO\nhandlers=console\n\n[logger_gunicorn.error]\nlevel=ERROR\nhandlers=console\npropagate=0\nqualname=gunicorn.error\n\n[handler_console]\nclass=StreamHandler\nformatter=json\nargs=(sys.stdout, )\n\n[formatter_json]\nclass=jsonlogging.JSONFormatter')

get_ipython().run_cell_magic('writefile', 'flaskwebapp/kill_supervisor.py', "import sys\nimport os\nimport signal\n\n\ndef write_stdout(s):\n    sys.stdout.write(s)\n    sys.stdout.flush()\n\n# this function is modified from the code and knowledge found here: http://supervisord.org/events.html#example-event-listener-implementation\ndef main():\n    while 1:\n        write_stdout('READY\\n')\n        # wait for the event on stdin that supervisord will send\n        line = sys.stdin.readline()\n        write_stdout('Killing supervisor with this event: ' + line);\n        try:\n            # supervisord writes its pid to its file from which we read it here, see supervisord.conf\n            pidfile = open('/tmp/supervisord.pid','r')\n            pid = int(pidfile.readline());\n            os.kill(pid, signal.SIGQUIT)\n        except Exception as e:\n            write_stdout('Could not kill supervisor: ' + e.strerror + '\\n')\n            write_stdout('RESULT 2\\nOK')\n\nmain()")

get_ipython().run_cell_magic('writefile', 'flaskwebapp/etc/supervisord.conf ', '[supervisord]\nlogfile=/tmp/supervisord.log ; (main log file;default $CWD/supervisord.log)\nlogfile_maxbytes=50MB        ; (max main logfile bytes b4 rotation;default 50MB)\nlogfile_backups=10           ; (num of main logfile rotation backups;default 10)\nloglevel=info                ; (log level;default info; others: debug,warn,trace)\npidfile=/tmp/supervisord.pid ; (supervisord pidfile;default supervisord.pid)\nnodaemon=true               ; (start in foreground if true;default false)\nminfds=1024                  ; (min. avail startup file descriptors;default 1024)\nminprocs=200                 ; (min. avail process descriptors;default 200)\n\n[program:gunicorn]\ncommand=bash -c "gunicorn --workers 1 -m 007 --timeout 100000 --capture-output --error-logfile - --log-config gunicorn_logging.conf \\"wsgi:create()\\""\ndirectory=/code\nredirect_stderr=true\nstdout_logfile =/dev/stdout\nstdout_logfile_maxbytes=0\nstartretries=2\nstartsecs=20\n\n[program:nginx]\ncommand=/usr/sbin/nginx -g "daemon off;"\nstartretries=2\nstartsecs=5\npriority=3\n\n[eventlistener:program_exit]\ncommand=python kill_supervisor.py\ndirectory=/code\nevents=PROCESS_STATE_FATAL\npriority=2')

get_ipython().run_cell_magic('writefile', 'flaskwebapp/dockerfile', '\nFROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04\nMAINTAINER Mathew Salvaris <mathew.salvaris@microsoft.com>\n\nRUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list\n\nRUN mkdir /code\nWORKDIR /code\nADD . /code/\nADD etc /etc\n\nRUN apt-get update && apt-get install -y --no-install-recommends \\\n        build-essential \\\n        ca-certificates \\\n        cmake \\\n        curl \\\n        git \\\n        nginx \\\n        supervisor \\\n        wget && \\\n        rm -rf /var/lib/apt/lists/*\n\nENV PYTHON_VERSION=3.5\nRUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \\\n    chmod +x ~/miniconda.sh && \\\n    ~/miniconda.sh -b -p /opt/conda && \\\n    rm ~/miniconda.sh && \\\n    /opt/conda/bin/conda create -y --name py$PYTHON_VERSION python=$PYTHON_VERSION numpy scipy pandas scikit-learn && \\\n    /opt/conda/bin/conda clean -ya\nENV PATH /opt/conda/envs/py$PYTHON_VERSION/bin:$PATH\nENV LD_LIBRARY_PATH /opt/conda/envs/py$PYTHON_VERSION/lib:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH\nENV PYTHONPATH /code/:$PYTHONPATH\n\nRUN rm /etc/nginx/sites-enabled/default && \\\n    cp /code/nginx/app /etc/nginx/sites-available/ && \\\n    ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/ && \\\n    pip install tensorflow-gpu==1.4.1 && \\\n    pip install -r /code/requirements.txt\n\nEXPOSE 80\nCMD ["supervisord", "-c", "/etc/supervisord.conf"]')

get_ipython().system('docker build -t $image_name -f $docker_file_location $application_path')

get_ipython().system("docker push $image_name # If you haven't loged in to the approrpiate dockerhub account you will get an error")

print('Docker image name {}'.format(image_name)) 

