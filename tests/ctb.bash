#!/usr/bin/env bash
set -e

id=$(docker run -tid --runtime=nvidia -P -v ~/dev/python/rl:/code celynw/esim_rl:v5)
infocmp | docker exec -i $id tic -x -o /usr/lib/terminfo /dev/stdin
# docker exec -it $id bash -ci "/code/venv/bin/python -m colored_traceback /code/ctb.py"
docker exec -it -e TERM $id bash -ci "/code/venv/bin/python /code/ctb.py"

# docker run -ti --runtime=nvidia -P -v ~/dev/python/rl:/code celynw/esim_rl:v4
