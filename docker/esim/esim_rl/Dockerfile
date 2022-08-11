FROM celynw/esim:v1 as esim
ARG USERNAME=docker

USER $root
RUN apt-get update -y && apt-get install -y --no-install-recommends \
	xvfb
USER $USERNAME

# Interactive (-i) session required for creating virtual environment properly...
SHELL ["/bin/bash", "-i", "-c"]
# Needs admin rights for writing to /opt/conda
# RUN conda update -n base -c defaults conda
RUN conda create -yp venv python=3.10 pip
RUN conda activate ./venv/ && \
	conda install -y pytorch torchvision cudatoolkit=11.3 -c pytorch
RUN conda activate ./venv/ && \
	conda install -y swig
# TODO use external file, e.g. requirements.txt to populate this
RUN conda activate ./venv/ && pip install \
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
	gym[all] \
	stable_baselines3 \
	msgpack \
	rospkg \
	wandb

# Cleanup
USER $root
RUN apt-get autoremove -y && \
	apt-get autoclean -y
RUN rm -rf /var/lib/apt/lists/*
# RUN conda clean --all -qvy
USER $USERNAME
RUN rm -r \
	~/.cache \
	~/.conda/pkgs
SHELL ["/bin/bash", "-c"]

# Onboarding
RUN mkdir -p /var/run/sshd
RUN echo "conda activate ~/venv" >> ~/.bashrc

STOPSIGNAL SIGINT
