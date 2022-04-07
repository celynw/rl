# RL for events

## Environment setup

```bash
sudo apt install ffmpeg freeglut3-dev xvfb
```

```bash
conda create -p venv python=3.10 pip
conda activate ./venv/
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
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
gym \
gym[atari] \
gym[box2d]
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
