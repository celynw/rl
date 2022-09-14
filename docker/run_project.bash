#!/usr/bin/env bash
set -e

# -S -s 500000 -d /runs/ --projection_head --optuna postgresql://postgres:password@$HOSTNAME:5432/postgres"
cmd="/vol/research/reflexive3/rl/venv/bin/python /vol/research/reflexive3/rl/toys/rgb_cartpole_ppo.py"
WANDB_API_KEY="local-10f17eeb44c533e126f6639891d16595a1c7f337"
WANDB_BASE_URL="http://umcvplws196:8080"

id=$(docker run -tid --runtime=nvidia -P -v /vol/research/reflexive3/rl/:/vol/research/reflexive3/rl/ -v /vol/research/reflexive3/rl/catkin_ws/:/vol/research/reflexive3/rl/catkin_ws/ -v /vol/research/reflexive3/rl/runs/:/vol/research/reflexive3/rl/runs/ -e WANDB_API_KEY=$WANDB_API_KEY -e WANDB_BASE_URL=$WANDB_BASE_URL esim_rl)
echo "Docker image ID: $id"

docker exec -id $id bash -ci "roscore"
docker exec -id $id bash -ci ". /vol/research/reflexive3/rl/catkin_ws/devel/setup.bash && roslaunch esim_ros --wait gym.launch config:=cfg/img.conf"
docker exec -i $id bash -ci "/vol/research/reflexive3/rl/venv/bin/python -m pip install -e /vol/research/reflexive3/rl"
docker exec -id $id bash -ci "Xvfb :0 -screen 0 800x600x24 +extension RANDR 2>/dev/null"
{
	echo "Running '$cmd $*'"
	docker exec -e DISPLAY=:0 -it $id bash -ci "$cmd $*"
} || {
	echo "Stopping and killing $id"
	docker logs -t $id
	docker kill $id
	docker rm $id
}
