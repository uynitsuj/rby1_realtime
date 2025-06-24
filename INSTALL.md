# Installation
Full install tested on Ubuntu 22.04

```
conda create -n rby python=3.11
conda activate rby
cd ~/
git clone --recurse-submodules https://github.com/uynitsuj/rby1_realtime.git
pip install -e .
pip install -e dependencies/robot_descriptions.py
pip install -e dependencies/pyroki
```
