#!/usr/bin/env bash
set -e
export WANDB_API_KEY="local-10f17eeb44c533e126f6639891d16595a1c7f337"
export WANDB_BASE_URL="http://umcvplws196:8080"
export DISPLAY=:0

/vol/research/reflexive3/rl/venv/bin/python -m pip install -e /vol/research/reflexive3/rl
source /vol/research/reflexive3/rl/catkin_ws/devel/setup.bash && roscore &
Xvfb :0 -screen 0 800x600x24 +extension RANDR 2>/dev/null &
source /vol/research/reflexive3/rl/catkin_ws/devel/setup.bash && roslaunch esim_ros --wait gym.launch config:=cfg/img.conf &
cmd="/vol/research/reflexive3/rl/venv/bin/python /vol/research/reflexive3/rl/toys/rgb_cartpole_ppo.py"
echo "Running '$cmd $*'"
source /vol/research/reflexive3/rl/catkin_ws/devel/setup.bash && $cmd $*
