# RL for events

## Environment setup

```bash
sudo apt install ffmpeg freeglut3-dev xvfb
```

```bash
conda create -p venv python=3.10 pip
conda activate ./venv/
conda install -y pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y swig
pip install \
pydocstyle \
mypy \
pycodestyle \
colored_traceback \
pytorch-lightning \
optuna \
rich \
gitpython \
setproctitle \
opencv-python \
gymnasium[accept-rom-license] \
stable-baselines3 \
msgpack \
rospkg \
wandb \
ale-py \
pyglet \
tensorboard \
moviepy \
pygame \
envpool
cd submodules/slayerPytorch
python setup.py install
cd -
# git+git://github.com/mila-iqia/atari-representation-learning.git \
pip install atari-representation-learning/
pip install -e .
```

gclr git@github.com:carlosluis/stable-baselines3.git
cd stable-baselines3/
gch fix_tests
	gch 5c32861

pip uninstall gym

Replace gym with gymnasium in the venv
- `import gym\n` -> `import gymnasium as gym\n`
- `from gym import ` -> `from gymnasium import `
- `from gym.` -> `from gymnasium.`

Fix wandb gym lazy imports `venv/lib/python3.10/site-packages/wandb/integration/gym/__init__.py`


## ROS setup

Go into a docker container.
Mount `catkin_ws`
Build catkin environment from inside (so all the paths work properly) (so we have to most up-to-date devel version of the repo)

sudo apt-get install ros-noetic-pybind11-catkin
pip install empy
catkin build esimcpp

melodic (fresh?):
	catkin build esimcpp --cmake-args -DPYTHON_VERSION=3.10 -DPYTHON_EXECUTABLE=/scratch/celyn/venv/bin/python
