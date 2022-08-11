environment = "mount=$ENV(PWD)"
executable = /vol/research/reflexive3/miniconda3/envs/pytorch/bin/python

# ---------------------------------------------------
# Universe (vanilla, docker)
universe = docker
docker_image = nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04

# -------------------------------------------------
# Event, out and error logs
log    = c$(cluster).p$(process).log
output = c$(cluster).p$(process).out
error  = c$(cluster).p$(process).error

# -----------------------------------
# File Transfer, Input, Output
should_transfer_files = YES


# -------------------------------------
# Requirements for the Job (see NvidiaDocker/Example09)
requirements = (HasDocker) && \
               (HasStornext) && \
               (TARGET.CondorPlatform == "$CondorPlatform: X86_64-Ubuntu_16.04 $") && \
               (CUDAGlobalMemoryMb > 4500) && (CUDAGlobalMemoryMb <  17000) && \
               (CUDACapability > 2.0)

# --------------------------------------
# Resources
request_GPUs   = 1
request_CPUs   = 1
request_memory = 4G

# ------------------------------------
# Request for guaruanteed run time. 0 means job is happy to checkpoint and move at any time.
# This lets Condor remove our job ASAP if a machine needs rebooting. Useful when we can checkpoint and restore
# Measured in seconds, so it can be changed to match the time it takes for an epoch to run
MaxJobRetirementTime = 0

# -----------------------------------
# Queue commands. We can use variables and flags to launch our command with multiple options (as you would from the command line)
arguments = $(script) --ckpt-dir $(ckpt_dir) --batch-size $(batch_size) --epochs $(epochs) --lr $(lr) --resume-training

# NOTE: Variable names can't contain dashes!
script = $ENV(PWD)/mnist_pytorch.py
ckpt_dir = $ENV(PWD)/models

batch_size = 32
epochs = 10
lr = 0.01

queue 1
