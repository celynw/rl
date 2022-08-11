from .modelType import ModelType

# TODO Do this automatically based on the environment definitions

discrete = [
	"CartPole-v0",
	# "CartPole-v1",
	"Acrobot-v1",
	"LunarLander-v2",
	"MountainCar-v0",
]
continuous = [
	"BipedalWalker-v3",
]
visual = [
	"Pong-v0",
	"CartPole-v1",
]

# ==================================================================================================
def choose_model(env: str):
	if env in discrete:
		return ModelType.DISCRETE
	elif env in visual:
		return ModelType.VISUAL
	else:
		raise ValueError(f"Environment '{env}' not yet supported")
