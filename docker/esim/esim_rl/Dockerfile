FROM celynw/esim:v5 as esim
ARG USERNAME=docker

ADD --chown=$USERNAME venv /home/docker/venv
ADD --chown=$USERNAME catkin_ws_local/devel/.private/esimcpp/lib/python3/dist-packages/esimcpp.cpython-310-x86_64-linux-gnu.so /home/docker/catkin_ws/devel/lib/python3/dist-packages/
WORKDIR /home/$USERNAME

ENV PYTHONPATH=/home/docker/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages
ENV LD_LIBRARY_PATH=/home/docker/catkin_ws/devel/lib:/home/docker/venv/lib:$LD_LIBRARY_PATH
# Then
#
# dr -ti --runtime=nvidia -v $HOME/Work/dev/py/rl:/home/docker/rl esim_rl
# dr -ti --gpus=all -v $HOME/Work/dev/py/rl:/home/docker/rl esim_rl
# podman run --user 1000:1000 --security-opt=no-new-privileges --cap-drop=ALL --security-opt label=type:nvidia_container_t docker.io/mirrorgooglecontainers/cuda-vector-add:v0.1
# podman run --user 1000:1000 --security-opt=no-new-privileges --cap-drop=ALL --security-opt label=type:nvidia_container_t celynw/esim_rl:v5
# podman run --user 1000:1000 --security-opt=no-new-privileges --cap-drop=ALL --security-opt label=type:nvidia_container_t celynw/esim_rl:v5

#
# ./venv/bin/python -m pip install -e rl/
# export WANDB_API_KEY=023a97fec4d807d5c9af2fd9418c9dccfa0cba5c
# ./venv/bin/python ./rl/tests/pong.py




# SHELL ["/opt/conda/condabin/conda", "run", "-p", "/home/docker/venv", "/bin/bash", "-c"]
# RUN python -m pip install .
# RUN /home/docker/venv/bin/python -m pip install .

# WORKDIR catkin_ws/

# # # ADD /home/celyn/Work/dev/py/rl/venv /home/docker/rl/venv
# # # docker build -t esim_rl --ssh default -f docker/esim/esim_rl/Dockerfile .

# # esimcpp
# RUN mkdir $HOME/.ssh
# RUN ssh-keyscan github.com >> $HOME/.ssh/known_hosts
# # # SSH forwarding usually only works as root, so specify our UID here
# RUN --mount=type=ssh,uid=1000 git clone --recursive git@github.com:celynw/esimcpp.git src/esimcpp
# # RUN pip install empy

# # RUN sudo apt-get update
# # RUN sudo apt-get install -y libpython3.8-dev
# # RUN sudo apt-get install python3-empy
# # RUN python -m pip install empy pybind11
# RUN python -m pip install empy
# # # RUN source /opt/conda/etc/profile.d/conda.sh && conda activate /home/docker/venv && catkin build --no-deps esimcpp
# # # RUN source /home/docker/catkin_ws/devel/setup.bash && source /opt/conda/etc/profile.d/conda.sh && conda activate /home/docker/venv && catkin build esimcpp
# # RUN source /home/docker/catkin_ws/devel/setup.bash && source /opt/conda/etc/profile.d/conda.sh && conda activate /home/docker/venv && catkin build esimcpp --cmake-args -DPYTHON_VERSION=3.10 -DPYTHON_EXECUTABLE=/home/$USERNAME/venv/bin/python
# RUN catkin build esimcpp
# # # RUN catkin build esimcpp --cmake-args -DPYTHON_VERSION=3.10 -DPYTHON_EXECUTABLE=/home/$USERNAME/venv/bin/python
# # # ENV PYTHONPATH=/home/cw0071/dev/python/rl/catkin_ws_local/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages

# WORKDIR /home/$USERNAME


# # # USER $root
# # # RUN apt-get update -y && apt-get install -y --no-install-recommends \
# # # 	xvfb
# # # USER $USERNAME

# # # # Cleanup
# # # USER $root
# # # RUN apt-get autoremove -y && \
# # # 	apt-get autoclean -y
# # # RUN rm -rf /var/lib/apt/lists/*
# # # # RUN conda clean --all -qvy
# # # USER $USERNAME
# # # SHELL ["/bin/bash", "-c"]

# # # # Onboarding
# # # RUN mkdir -p /var/run/sshd

STOPSIGNAL SIGINT
