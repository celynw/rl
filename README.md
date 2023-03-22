# RL for events

## Environment setup

```bash
sudo apt install ffmpeg freeglut3-dev xvfb
```

```bash
conda create -yp venv python=3.10 pip
conda activate ./venv/

conda install -yc pytorch -c nvidia \
pytorch \
torchvision \
pytorch-cuda=11.7

conda install -y \
swig \
pytorch-lightning \
rich \
gitpython \
setproctitle \
gymnasium[accept-rom-license] \
tensorboard

conda install -yc conda-forge \
stable-baselines3 \
rospkg \
wandb \
ale-py \
moviepy \
pygame \
pyglet \
opencv \
pydocstyle \
mypy \
pycodestyle \
colored-traceback \
optuna \
msgpack-python

pip install tcod

pip install git+https://github.com/mila-iqia/atari-representation-learning

cd submodules/slayerPytorch
python setup.py install

pip uninstall stable-baselines3
pip install git+https://github.com/DLR-RM/stable-baselines3@feat/gymnasium-support
pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib@feat/gymnasium-support
pip uninstall gym

pip install -e .
```

## ROS setup

Go into a docker container.
Mount `catkin_ws`
Build catkin environment from inside (so all the paths work properly) (so we have to most up-to-date devel version of the repo)

sudo apt-get install ros-noetic-pybind11-catkin
pip install empy
catkin build esimcpp

melodic (fresh?):
	catkin build esimcpp --cmake-args -DPYTHON_VERSION=3.10 -DPYTHON_EXECUTABLE=/scratch/celyn/venv/bin/python
