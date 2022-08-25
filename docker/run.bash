#!/usr/bin/env bash
set -e

# cmd="/code/toys/rgb_cartpole_ppo.py 1 -S -s 200000 -d /runs/"
# cmd="/code/venv/bin/python /code/toys/rgb_cartpole_ppo.py 1 -S -s 200000 -d /runs/ --lr 3e-4 --lr2 3e-3"
# cmd="/code/venv/bin/python /code/toys/rgb_cartpole_ppo.py 1 -S -s 200000 -d /runs/"
cmd="/code/venv/bin/python /code/toys/rgb_cartpole_ppo.py -S -s 200000 -d /runs/ --optuna postgresql://postgres:password@$HOSTNAME:5432/postgres"
WANDB_API_KEY="local-10f17eeb44c533e126f6639891d16595a1c7f337"
WANDB_BASE_URL="http://umcvplws196:8080"

id=$(docker run -tid --runtime=nvidia -P -v ~/dev/python/rl/:/code/ -v ~/dev/python/rl/catkin_ws/:/catkin_ws/ -v ~/dev/python/rl/runs/:/runs/ -e WANDB_API_KEY=$WANDB_API_KEY -e WANDB_BASE_URL=$WANDB_BASE_URL esim_rl)
echo "Docker image ID: $id"

docker exec -id $id bash -ci "roscore"
docker exec -id $id bash -ci ". /catkin_ws/devel/setup.bash && roslaunch esim_ros --wait gym.launch config:=cfg/img.conf"
# docker exec -i $id bash -ci "pip install -e /code"
docker exec -i $id bash -ci "/code/venv/bin/python -m pip install -e /code"
docker exec -id $id bash -ci "Xvfb :0 -screen 0 800x600x24 +extension RANDR 2>/dev/null"
{
	if [ "$#" -eq 0 ]; then
		echo "Expected name argument"
	elif [ "$#" -eq 1 ]; then
		docker exec -e DISPLAY=:0 -it $id bash -ci "$cmd $1"
	else
		echo "Too many arguments!"
	fi
} || {
	echo "Stopping and killing $id"
	docker logs -t $id
	docker kill $id
	docker rm $id
}
