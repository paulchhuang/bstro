## Installation

Our codebase is developed based on Ubuntu 18.04 and NVIDIA GPU cards. 

### Requirements
- Python 3.6.9
- Pytorch 1.4
- torchvision 0.5.0
- cuda 10.1

### Setup environment

```bash
# Create a new environment
python3 -m venv PATH/2/VENV
source PATH/2/VENV/bin/activate

# Install Pytorch
pip install torch==1.4.0 torchvision==0.5.0 

export INSTALL_DIR=$PWD

# Install METRO
cd $INSTALL_DIR
git clone --recursive https://github.com/paulchhuang/bstro.git
cd bstro
python setup.py build develop

# Install requirements
pip install -r requirements.txt

unset INSTALL_DIR
```


