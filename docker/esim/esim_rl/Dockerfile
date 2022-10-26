FROM celynw/esim:v5 as esim
ARG USERNAME=docker

USER $root
RUN apt-get update -y && apt-get install -y --no-install-recommends \
	xvfb
USER $USERNAME

# Cleanup
USER $root
RUN apt-get autoremove -y && \
	apt-get autoclean -y
RUN rm -rf /var/lib/apt/lists/*
# RUN conda clean --all -qvy
USER $USERNAME
SHELL ["/bin/bash", "-c"]

# Onboarding
RUN mkdir -p /var/run/sshd

STOPSIGNAL SIGINT
