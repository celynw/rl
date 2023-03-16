import numpy as np

# ==================================================================================================
def get_base_envs(env, avoidList: bool = False) -> list:
	# env.unwrapped doesn't get past DummyVecEnv
	if hasattr(env, "envs"):
		return [get_base_envs(e, avoidList=True) for e in env.envs]
	else:
		if hasattr(env, "env"):
			return get_base_envs(env.env, avoidList=avoidList)
		else:
			if avoidList:
				return env
			else:
				return [env]

# ==================================================================================================
def add_event_image_channel(img):
	return np.concatenate((img, np.zeros_like(img)[:1]), axis=0)
