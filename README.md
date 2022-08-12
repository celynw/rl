# RL for events

## Environment setup

```bash
sudo apt install ffmpeg freeglut3-dev xvfb
```

```bash
conda create -p venv python=3.10 pip
conda activate ./venv/
conda install -y pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install -y swig
pip install \
pydocstyle \
mypy \
pycodestyle \
colored_traceback \
pytorch-lightning \
optuna \
rich \
kellog \
gitpython \
setproctitle \
opencv-python \
# h5py \
gym[all]
stable-baselines3 \
msgpack \
rospkg
```

(For Stable Baselines):

```bash
pip install \
stable-baselines3[extra] \
pyglet
```

(For Nengo):

```bash
pip install \
nengo-gui \
nengo \
jedi \
ipykernel
```

## ROS setup

Go into a docker container.
Mount `catkin_ws`
Build catkin environment from inside (so all the paths work properly) (so we have to most up-to-date devel version of the repo)
