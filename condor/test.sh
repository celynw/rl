# ==================================================================================================
universe = docker
docker_image = celynw/esim_rl:v3
environment = "mount=$(_ws)"

# Requirements -------------------------------------------------------------------------------------
request_GPUs = 1
request_CPUs = 1
request_memory = 8G
# requirements = (HasDocker) && \
# 			   (HasStornext) && \
# 			   (TARGET.CondorPlatform == "$CondorPlatform: X86_64-Ubuntu_16.04 $") && \
# 			   (CUDAGlobalMemoryMb > 4500) && (CUDAGlobalMemoryMb < 17000) && \
# 			   (CUDACapability > 2.0)
requirements = (HasDocker)\
	&& (machine != "sounds01.eps.surrey.ac.uk")\
	&& (machine != "gloin.eps.surrey.ac.uk")\
	&& (machine != "fili.eps.surrey.ac.uk")\
	&& (machine != "nimrodel.eps.surrey.ac.uk")\
	&& (machine != "dwalin.eps.surrey.ac.uk")\
	&& (machine != "bofur.eps.surrey.ac.uk")\
	&& (machine != "cogvis2.eps.surrey.ac.uk")\
	&& (CUDAGlobalMemoryMb > 4000)
	# && (CUDACapability > 2.0)\
	# && (machine != "aisurrey06.surrey.ac.uk")\
	# && (machine != "aisurrey03.surrey.ac.uk")\
	# && (machine != "aisurrey15.surrey.ac.uk")\

# Bad:
# - sounds01: Unable to find image 'celynw/esim_rl:v3' locally
# - gloin: Unable to find image 'celynw/esim_rl:v3' locally
# - fili: Unable to find image 'celynw/esim_rl:v3' locally
# - nimrodel: Unable to find image 'celynw/esim_rl:v3' locally
# - dwalin: Unable to find image 'celynw/esim_rl:v3' locally
# - bofur: Unable to find image 'celynw/esim_rl:v3' locally
# - cogvis2: Keeps getting evicted
# Good:
# - balin

# Variables ----------------------------------------------------------------------------------------
_ws = /vol/research/reflexive3
_wd = $(_ws)/rl
_name = cartPole_EDeNN_grid
# _optuna = --optuna postgresql://postgres:password@umcvplws196.surrey.ac.uk:5432/postgres
_optuna =
_steps = -s 500000
_ph = --projection_head
_out = -d $(_wd)/runs/

# Command ------------------------------------------------------------------------------------------
executable = $(_wd)/docker/run_condor.bash
arguments = $(_name) -S $(_out) $(_optuna) $(_steps) $(_ph) --tsamples $(_tsamples)

# Other parameters ---------------------------------------------------------------------------------
should_transfer_files = YES
log = $(_wd)/runs/condor/c-$(cluster)_p-$(process).log
output = $(_wd)/runs/condor/c-$(cluster)_p-$(process).out
error = $(_wd)/runs/condor/c-$(cluster)_p-$(process).error
# Request for guaruanteed run time. 0 means job is happy to checkpoint and move at any time.
# This lets Condor remove our job ASAP if a machine needs rebooting. Useful when we can checkpoint and restore
# Measured in seconds, so it can be changed to match the time it takes for an epoch to run
# MaxJobRetirementTime = 0
+CanCheckpoint = false
# JobRunTime is in hours
+JobRunTime = 8
+GPUMem = 8000
# stream_output = true

# Queue ============================================================================================
queue _tsamples from (
	1
	2
	3
	4
	5
	6
	7
	8
	9
	10
)
