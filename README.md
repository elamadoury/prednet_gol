# Prednet install:

## Reference
cloned code at https://github.com/coxlab/prednet
ref https://coxlab.github.io/prednet/

## Installation

Switch python versions: 
`> python --version
Python 3.4.2
> set PATH=C:\tools\python2\;%PATH%
> python --version
Python 2.7.9`

### Install python, pip and virtualenv according to your OS 
python, pip and virtualenv install manual: https://www.tensorflow.org/install/pip

Example for windows:
Go to Visual Studio https://visualstudio.microsoft.com/vs/older-downloads/
Select Redistributables and Build Tools,
Download and install the Microsoft Visual C++ 2015 Redistributable Update 3.

Go to https://www.python.org/downloads/windows/
Install the 64-bit Python 3 release for Windows (3.6), select pip as an optional feature.

Install virtualenv through a new command prompt (run as administrator)
`pip3 install -U pip virtualenv`
or
`pip install -U pip virtualenv`

### Install Tensorflow
Tensorflow install manual: https://www.tensorflow.org/install/

Create a virtual environment
`virtualenv --system-site-packages -p py ./path_to_your_venv`
Activate env and install packages
`path_to_your_venv\Scripts\activate`
`pip install --upgrade pip`
For CPU tensorflow install
`pip install --upgrade tensorflow` (`pip install --upgrade tensorflow-gpu` for gpu)


### Install Keras
Keras install manual : https://keras.io/#installation

From your virtual env
`pip install keras`

## Setup Prednet

Install packages
`pip install requests bs4 imageio scipy hickle matplotlib`

### hkl pickle error:
Install python2.7 and make a virtual env
Install old hickle
`pip install hickle==3.2.1`
`cd fix_prednet_data`
`py hkl_py_py2.py`

Go back to python3 venv
`pip install hickle==3.3.2`
`py hkl_py2_py3.py`

change hkl read names in approprite files, eg in kitti_evaluate.py `X_test.hkl` -> `X_test_36.hkl`

(ref: https://stackoverflow.com/questions/51413618/loading-hickle-filecomes-from-python2-in-python-3
https://github.com/telegraphic/hickle)

Install wget and unzip how you can (OS dependent)
Download data
`cd prednet_dir`
`py process_kitti.py` or faster, `sh download_data.sh` or download directly from https://www.dropbox.com/s/rpwlnn6j39jjme4/kitti_data.zip?dl=0

The file

## Test

run the model `py kitti_evaluate.py`

### GPU setup (Windows)

References: 
https://www.tensorflow.org/install/pip
https://www.tensorflow.org/install/gpu
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/

Install tensorflow-gpu 
`pip install --upgrade tensorflow-gpu`

Install Nvidia drivers. Find your GPU model (https://www.cisco.com/c/en/us/td/docs/telepresence/endpoint/articles/cisco_telepresence_movi_find_out_graphics_card_driver_on_windows_pc_kb_540.html)
eg NVDIA GeForce RTX 2080 Ti
Download corresponding driver https://www.nvidia.com/Download/index.aspx?lang=en-us
Install CUDA toolkit https://developer.nvidia.com/cuda-zone
