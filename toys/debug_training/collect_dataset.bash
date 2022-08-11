#!/usr/bin/env bash
set -e

id=$(docker run -tid --runtime=nvidia -P -v ~/dev/python/rl/:/code/ -v ~/dev/python/rl/catkin_ws/:/catkin_ws/ -v ~/dev/python/rl/runs/:/runs/ esim_rl)
echo "Docker image ID: $id"
docker exec -id $id bash -ci "roscore"
docker exec -id $id bash -ci ". /catkin_ws/devel/setup.bash && roslaunch esim_ros --wait gym.launch config:=cfg/img.conf"
docker exec -i $id bash -ci "pip install -e /code"
docker exec -id $id bash -ci "Xvfb :0 -screen 0 800x600x24 +extension RANDR 2>/dev/null"
{
	docker exec -e DISPLAY=:0 -it $id bash -ci "/code/toys/debug_training/collect_dataset.py 10000"
} || {
	echo "Stopping and killing $id"
	docker logs -t $id
	docker kill $id
	docker rm $id
}
