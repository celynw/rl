#!/usr/bin/env bash
set -e
cmd="/code/venv/bin/python /code/"

id=$(docker run -tid --runtime=nvidia -P -v ~/dev/python/rl/venv:/home/docker/venv -v ~/dev/python/rl:/code -v ~/dev/python/rl/catkin_ws:/catkin_ws -v ~/dev/python/rl/runs:/runs -e WANDB_BASE_URL -e WANDB_API_KEY -e TERM celynw/esim_rl:v5)
echo "Docker image ID: $id"

infocmp | docker exec -i $id tic -x -o /usr/lib/terminfo /dev/stdin # Allow user's terminfo for their current TERM
docker exec -id $id bash -ci "roscore"
docker exec -id $id bash -ci ". /catkin_ws/devel/setup.bash && roslaunch esim_ros --wait gym.launch config:=cfg/img.conf"
# docker exec -it $id bash -ci "/code/venv/bin/python -m pip install -e /code"
docker exec -id $id bash -ci "Xvfb :0 -screen 0 800x600x24 +extension RANDR 2>/dev/null"
{
	echo "Running '$cmd$*'"
	docker exec -e DISPLAY=:0 -it $id bash -ci "$cmd$*"
} || {
	echo "Stopping and killing $id"
	docker logs -t $id
	docker kill $id
	docker rm $id
}
