#!/usr/bin/env python3
import argparse
import dataclasses

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from gym import spaces
from slayerSNN.slayer import spikeLayer
from rich import print, inspect

from rl.models.utils import MetaTensor, TensorLayout, DataType, NeuronConfig

# ==================================================================================================
class SNN(BaseFeaturesExtractor):
	_input_key: str = "input"
	_output_key: str = "output"
	# ----------------------------------------------------------------------------------------------
	def __init__(self, observation_space: spaces.Box, features_dim: int, fps: float, tsamples: int):
		super().__init__(observation_space, features_dim)
		self.observation_space = observation_space

		simulation_params = {
			"Ts": ((1 / fps) / tsamples) * 1e3, # Length of time for each bin (ms)
			"tSample": (1 / fps) * 1e3, # Length of time for the window (ms)
		}
		self.init_layers(simulation_params)
		self._data = dict()

	# ----------------------------------------------------------------------------------------------
	@staticmethod
	def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
		"""
		Appends model-specific arguments to the parser.

		Args:
			parser (argparse.ArgumentParser): Main parser object.

		Returns:
			argparse.ArgumentParser: Modified parser object.
		"""
		group = parser.add_argument_group("Model")

		return parser

	# ----------------------------------------------------------------------------------------------
	def init_layers(self, simulation_params: dict):
		neuron_config_conv1 = dataclasses.asdict(NeuronConfig(
			theta=0.27,
			tauSr=2.0,
			tauRef=1.0,
			scaleRho=0.20,
		))
		neuron_config_conv2 = dataclasses.asdict(NeuronConfig(
			theta=0.25,
			tauSr=2.0,
			tauRef=1.0,
			scaleRho=0.13,
		))
		neuron_config_conv3 = dataclasses.asdict(NeuronConfig(
			theta=0.3,
			tauSr=4.0,
			tauRef=4.0,
			scaleRho=0.13,
		))
		# neuron_config_conv4 = dataclasses.asdict(NeuronConfig(
		# 	theta=0.4,
		# 	tauSr=4.0,
		# 	tauRef=4.0,
		# 	scaleRho=0.25,
		# ))
		# neuron_config_conv5 = dataclasses.asdict(NeuronConfig(
		# 	theta=0.4,
		# 	tauSr=4.0,
		# 	tauRef=4.0,
		# 	scaleRho=100.0,
		# ))
		# NOTE: tauSr is the only parameter that has any influence on the output or the gradients
		neuron_config_fc = dataclasses.asdict(NeuronConfig(tauSr=8.0))

		self.slayer_conv1 = spikeLayer(neuron_config_conv1, simulation_params)
		self.slayer_conv2 = spikeLayer(neuron_config_conv2, simulation_params)
		self.slayer_conv3 = spikeLayer(neuron_config_conv3, simulation_params)
		self.slayer_fc = spikeLayer(neuron_config_fc, simulation_params)

		conv1_out_channels = 32
		conv2_out_channels = 64
		conv3_out_channels = 128

		self.conv1 = self.slayer_conv1.conv(
			inChannels=2, outChannels=conv1_out_channels,
			kernelSize=8, stride=4, padding=0, dilation=1, groups=1, weightScale=1,
		)
		self.conv2 = self.slayer_conv2.conv(
			inChannels=conv1_out_channels, outChannels=conv2_out_channels,
			kernelSize=4, stride=2, padding=0, dilation=1, groups=1, weightScale=1,
		)
		self.conv3 = self.slayer_conv3.conv(
			inChannels=conv2_out_channels, outChannels=conv3_out_channels,
			kernelSize=3, stride=1, padding=0, dilation=1, groups=1, weightScale=1,
		)
		self.fc = self.slayer_fc.dense(inFeatures=conv3_out_channels, outFeatures=self.features_dim, weightScale=1)

	# ----------------------------------------------------------------------------------------------
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# NOTE: spikeLayer.conv expects tensors in NCHWT. Need to permute from standard BCDHW
		x = x.permute(0, 1, 3, 4, 2) # BCDHW -> NCHWT

		self.addInputMetaTensor(MetaTensor(x, TensorLayout.Conv, DataType.Spike))

		spikes_layer_1 = self.slayer_conv1.spike(self.conv1(self.slayer_conv1.psp(x)))
		spikes_layer_2 = self.slayer_conv2.spike(self.conv2(self.slayer_conv2.psp(spikes_layer_1)))
		spikes_layer_3 = self.slayer_conv3.spike(self.conv3(self.slayer_conv3.psp(spikes_layer_2)))

		self.addMetaTensor("conv1", MetaTensor(spikes_layer_1, TensorLayout.Conv, DataType.Spike))
		self.addMetaTensor("conv2", MetaTensor(spikes_layer_2, TensorLayout.Conv, DataType.Spike))
		self.addMetaTensor("conv3", MetaTensor(spikes_layer_3, TensorLayout.Conv, DataType.Spike))

		# Apply average pooling on spike-trains.
		spikes_mean = torch.mean(spikes_layer_3, dim=(2, 3), keepdims=True) # type: ignore
		psp_out = self.slayer_fc.psp(self.fc(spikes_mean))
		self.addOutputMetaTensor(MetaTensor(psp_out, TensorLayout.FC, DataType.Dense))

		assert psp_out.shape[1] == self.features_dim

		psp_out = psp_out.permute(0, 1, 4, 2, 3) # NCHWT -> BCDHW
		psp_out = psp_out[:, :, -1] # FIX Is it really OK to just take the last time bin?

		return psp_out.squeeze(-1).squeeze(-1) # [1, self.features_dim]

	# ----------------------------------------------------------------------------------------------
	def addMetaTensor(self, key: str, value: MetaTensor):
		assert not key == self._output_key, "Use addOutputMetaTensor function instead"
		self._data[key] = value

	# ----------------------------------------------------------------------------------------------
	def addInputMetaTensor(self, value: MetaTensor):
		assert value.isSpikeType()
		assert value.hasConvLayout(), "Does not have to be but is reasonable for the moment"
		self._data[self._input_key] = value

	# ----------------------------------------------------------------------------------------------
	def addOutputMetaTensor(self, value: MetaTensor):
		assert value.isDenseType()
		assert value.hasFCLayout(), "Does not have to be but is reasonable for the moment"
		self._data[self._output_key] = value


# ==================================================================================================
if __name__ == "__main__":
	import gym
	from rich import print, inspect

	import rl
	import rl.environments

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser = SNN.add_argparse_args(parser)
	parser = rl.environments.CartPoleEvents.add_argparse_args(parser)
	args = parser.parse_args()

	fps = 30
	args.tsamples = 100

	env = gym.make(
		"CartPoleEvents-v0",
		args=args,
	)
	snn = SNN(observation_space=env.observation_space, features_dim=env.state_space.shape[-1], fps=fps, tsamples=args.tsamples)

	event_tensor = torch.rand([1, 2, args.tsamples, env.output_height, env.output_width])
	print(f"input event shape: {event_tensor.shape}")
	print(f"features_dim: {snn.features_dim}")

	snn = snn.to("cuda")
	event_tensor = event_tensor.to("cuda")
	output = snn(event_tensor)

	print(f"model: {snn}")
	print(f"layer 1: {snn._data['conv1'].getTensor().shape}")
	print(f"layer 2: {snn._data['conv2'].getTensor().shape}")
	print(f"layer 3: {snn._data['conv3'].getTensor().shape}")
	print(f"output features shape: {output.shape}")
