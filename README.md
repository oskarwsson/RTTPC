# RTTPC
Robot terrain traversability planning using only a camera and vision foundation models

Project in the course DD2600 at KTH Stockholm.

## Installation/setup
In the code repository, create a virtual environment by running: 
```bash
uv venv --python 3.12 .venv
```

To enter the environment, run: 
```bash
source .venv/bin/activate
```
Also add it to the known kernels:
```bash
.venv/bin/python -m ipykernel install --user --name rttpc --display-name "Python (RTTPC)"
```

Finally, to install dependencies, run: 
```bash
uv sync
```  

Done!

For faster loading of models, make sure you have set up a HuggingFace access token (``HF_TOKEN``) in your system variables.  

### Open3D issue
If you for some reason are missing the libEGL library on your system, and you lack the permissions to install it system-wide, you can follow these steps to install it at user-level. 

Download the libraries:
```bash
mkdir -p "$HOME/.local/open3d-libs"
(
    cd "$HOME/.local/open3d-libs" || exit
    apt-get download libegl1 libglvnd0
    for package in *.deb; do
        dpkg-deb -x "$package" .
    done
)
```

Then, from the RTTPC directory, you run this:
```bash
export LD_LIBRARY_PATH="$HOME/.local/open3d-libs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

To verify that Open3D works, you can run the following:
```bash
.venv/bin/python -c "import open3d; print(open3d.__version__)"
```

If you want this to work for notebooks, you need to add it to the kernel. Run this to find the kernel location:
```bash
jupyter kernelspec list
```
Then, in the listed directory, find ``kernel.json`` and add this field:
```json
"env": {
  "LD_LIBRARY_PATH": "YOUR PATH HERE"
}
```
To find the path above, you can run
```bash
echo $HOME/.local/open3d-libs/usr/lib/x86_64-linux-gnu
```
and copy the output. It should look something like this in the end:
```json
"env": {
  "LD_LIBRARY_PATH": "/home/jovyan/.local/open3d-libs/usr/lib/x86_64-linux-gnu"
}
```
(but with your username, of course)
## Running
Just do it