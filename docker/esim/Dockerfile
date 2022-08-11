FROM ros:noetic-perception-focal as ros

# ROS
# Create a user
ARG USERNAME=docker
RUN mkdir -p /home/$USERNAME
WORKDIR /home/$USERNAME
# ESIM
RUN apt-get update -y && apt-get install -y --no-install-recommends \
	git
RUN sudo apt-get install -y --no-install-recommends \
	ros-noetic-pcl-ros \
	libproj-dev \
	libglfw3 libglfw3-dev \
	libglm-dev \
	ros-noetic-hector-trajectory-server \
	ros-noetic-tf-conversions \
	python3-catkin-tools \
	libtool
RUN mkdir $HOME/.ssh
RUN ssh-keyscan github.com >> $HOME/.ssh/known_hosts
RUN mkdir -p catkin_ws/src
WORKDIR catkin_ws/
RUN catkin init
RUN catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-Wno-int-in-bool-context
WORKDIR src/
RUN --mount=type=ssh git clone --recursive git@github.com:celynw/rpg_esim.git
WORKDIR rpg_esim/
RUN rm -rf .git
WORKDIR submodules/ze_oss/
RUN touch \
	imp_3rdparty_cuda_toolkit/CATKIN_IGNORE \
	imp_app_pangolin_example/CATKIN_IGNORE \
	imp_benchmark_aligned_allocator/CATKIN_IGNORE \
	imp_bridge_pangolin/CATKIN_IGNORE \
	imp_cu_core/CATKIN_IGNORE \
	imp_cu_correspondence/CATKIN_IGNORE \
	imp_cu_imgproc/CATKIN_IGNORE \
	imp_ros_rof_denoising/CATKIN_IGNORE \
	imp_tools_cmd/CATKIN_IGNORE \
	ze_data_provider/CATKIN_IGNORE \
	ze_geometry/CATKIN_IGNORE \
	ze_imu/CATKIN_IGNORE \
	ze_trajectory_analysis/CATKIN_IGNORE
WORKDIR ../../../../
RUN catkin build esim_ros

# Cleanup
RUN apt-get autoremove -y && \
	apt-get autoclean -y
RUN rm -rf /var/lib/apt/lists/*

# NVIDIA
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04
# Create a user
ARG USERNAME=docker
ARG USERID=1000
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
	adduser $USERNAME sudo && \
	echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
WORKDIR /home/$USERNAME
# /usr/bin/, /usr/lib/, /usr/share/
COPY --from=ros /usr/ /usr/
COPY --from=ros /lib/ /lib/
COPY --from=ros /etc/alternatives/ /etc/alternatives/
COPY --from=ros /opt/ros/ /opt/ros/
COPY --from=ros /home/$USERNAME/catkin_ws/ /home/$USERNAME/catkin_ws/

# Miniconda
ARG CONDA_DIR=/opt/conda
RUN apt-get update && apt-get install -y --no-install-recommends \
	python-is-python3 \
	wget
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
	rm -rf /tmp/*

# Cleanup
RUN apt-get autoremove -y && \
	apt-get autoclean -y
RUN rm -rf /var/lib/apt/lists/*

# Onboarding
RUN mkdir -p /var/run/sshd
USER $USERNAME
RUN echo ". /opt/ros/noetic/setup.bash" >> /home/$USERNAME/.bashrc
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/$USERNAME/.bashrc
RUN echo ". ~/catkin_ws/devel/setup.bash" >> /home/$USERNAME/.bashrc
SHELL ["/bin/bash", "-c"]

STOPSIGNAL SIGINT
