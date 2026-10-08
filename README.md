# RTTPC
Robot terrain traversability planning using only a camera and vision foundation models

Project in the course DD2600 at KTH Stockholm.

## Installation/setup
In the code repository, create a virtual environment by running: 
```
uv venv --python 3.12 .venv
```

To enter the environment, run: 
```
source .venv/bin/activate
```

Finally, to install dependencies, run: 
```
uv sync
```  

Done!

For faster loading of models, make sure you have set up a HuggingFace access token (``HF_TOKEN``) in your system variables.  