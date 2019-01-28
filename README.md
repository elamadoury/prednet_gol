# Prednet install:

## Reference
cloned code at https://github.com/coxlab/prednet
ref https://coxlab.github.io/prednet/

## Installation


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
