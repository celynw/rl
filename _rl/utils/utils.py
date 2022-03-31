#!/usr/bin/env python3
from enum import Enum
import argparse
from pathlib import Path
import inspect
import time
import sys
import textwrap
from typing import Tuple, List, Union

import torch
from git import Repo
import setproctitle
from kellog import info, warning, error, debug
import numpy as np
import cv2
from pytorch_lightning import LightningDataModule, loggers
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint, LearningRateMonitor, RichProgressBar
from typing import Optional, Callable
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from optuna.trial import Trial
import kellog
from kellog import info, warning, error, debug

import rl

eps = 1e-15
# eps = torch.finfo(tensor.dtype).eps

# ==================================================================================================
class Step(Enum):
	TRAIN = "train"
	VAL = "val"
	TEST = "test"
	# ----------------------------------------------------------------------------------------------
	def __str__(self):
		return self.value


# ==================================================================================================
class DataModule(LightningDataModule):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, cpu_only: List[str] = []):
		super().__init__()
		self.cpu_only = cpu_only

	# ----------------------------------------------------------------------------------------------
	def transfer_batch_to_device(self, batch, device, dataloader_idx):
		if isinstance(batch, dict):
			for key in batch:
				if key not in self.cpu_only:
					batch[key] = super().transfer_batch_to_device(batch[key], device, dataloader_idx)

			return batch
		else:
			return super().transfer_batch_to_device(batch, device, dataloader_idx)


# ==================================================================================================
class ArgumentParser(argparse.ArgumentParser):
	"""Override argparse.ArgumentParser."""
	# ----------------------------------------------------------------------------------------------
	def error(self, message: str):
		"""
		Print the problems with the arguments.
		Also prints a helpful message about model types.

		Args:
			message (str): Message to print
		"""
		self.print_usage(sys.stderr)
		args = {"prog": self.prog, "message": message}
		self.exit(2, textwrap.dedent((f"""\
			{args['prog']}: error: {args['message']}
			More help options will be provided when specifying the model type
		""")))


# ==================================================================================================
def parse_args() -> argparse.Namespace:
	models = [name for name, obj in inspect.getmembers(rl.models) if inspect.isclass(obj) and name not in ["Base", "RL"]]

	# Help argument is ignored until we have added the args for the other modules!
	parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
	group = parser.add_argument_group("Trainer")
	group.add_argument("model", choices=models, metavar=f"MODEL: {{{', '.join(models)}}}", help="Model to use")
	args_ = parser.parse_known_args()

	# NOW add the help...
	parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")
	group.add_argument("-o", "--output_directory", type=Path, help="Output directory")
	group.add_argument("--lr", type=float, default=0.01, help="Learning rate")
	group.add_argument("-v", "--version", type=str, help="Try to continue training from this version", default=None)
	group.add_argument("--cpu", action="store_true", help="Use CPU rather than GPU as device")
	group.add_argument("--optuna", action="store_true", help="Optimise with optuna")
	group.add_argument("--profile", action="store_true", help="Profile a single epoch with tensorboard")
	group.add_argument("--no_graph", action="store_true", help="Disable tensorboard graph logging (sometimes it just won't play nice)")
	group.add_argument("--seed", type=int, help="Use specified random seed for everything", default=None)
	group.add_argument("--train_lim", type=float, help="Only train on this many batches, or this proportion", default=1.0)
	group.add_argument("--val_lim", type=float, help="Only validate on this many batches, or this proportion", default=1.0)
	group.add_argument("--test_lim", type=str, help="Use this test proportion (float) or batches (int) each epoch (still randomised over entire dataset)", default="1.0")
	group.add_argument("--overfit", type=str, help="Overfit to this proportion (float) or batches (int), use train set for val", default="0.0")
	group.add_argument("-C", "--checkpoint", type=Path, help="Path to checkpoint file. Can be relative to specific model log directory", default="last.ckpt")
	import __main__
	group.add_argument("--proctitle", type=str, help="Process title", default=Path(__main__.__file__).name)

	Model = getattr(rl.models, args_[0].model)
	parser = Model.add_argparse_args(parser)

	args = parser.parse_args()

	args.train_lim = limit_float_int(args.train_lim)
	args.val_lim = limit_float_int(args.val_lim)
	args.test_lim = limit_float_int(args.test_lim)
	args.overfit = limit_float_int(args.overfit)
	try:
		args.version = int(args.version)
	except:
		pass

	setproctitle.setproctitle(args.proctitle)

	return args


# ==================================================================================================
def get_called_command():
	import sys
	return " ".join(sys.argv)


# ==================================================================================================
def get_gpu_info():
	gpus = []
	for gpu in range(torch.cuda.device_count()):
		properties = torch.cuda.get_device_properties(f"cuda:{gpu}")
		gpus.append({
			"name": properties.name, # type: ignore
			"memory": round(properties.total_memory / 1e9, 2), # type: ignore
			"capability": f"{properties.major}.{properties.minor}", # type: ignore
		})

	return gpus


#===================================================================================================
def get_git_rev(cwd=Path(inspect.stack()[1][1]).parent) -> str: # Uses parent of called script by default
	repo = Repo(cwd)
	sha = repo.head.commit.hexsha
	output = repo.git.rev_parse(sha, short=7)
	if repo.is_dirty():
		output += " (dirty)"
	output += " - " + time.strftime("%a %d/%m/%Y %H:%M", time.gmtime(repo.head.commit.committed_date))

	return output


# ==================================================================================================
def get_checkpoint(logger: loggers.TensorBoardLogger, checkpoint: Optional[Union[str, Path]] = None) -> Optional[Path]:
	if logger is None:
		checkpointPath = None
	else:
		ckptDir = Path(logger.log_dir) / "checkpoints"
		checkpointPath = ckptDir / checkpoint
		if not checkpointPath.exists():
			checkpointPath = checkpoint
		if not checkpointPath.exists():
			if checkpoint is not None:
				warning(f"Specified checkpoint not found at '{checkpointPath}' does not exist!")
			checkpointPath = None

		return checkpointPath


# ==================================================================================================
def setup_logger(args) -> Optional[loggers.TensorBoardLogger]:
	if args.output_directory is None:
		warning("No output directory specified, will not log!")
		return None
	else:
		try:
			args.version = int(args.version)
		except:
			pass
		logger = loggers.TensorBoardLogger(
			name=args.model,
			save_dir=args.output_directory,
			log_graph=not args.no_graph,
			version=args.version,
			default_hp_metric=False,
		)
		# Would be created later, but we need it now for kellog logs
		Path(logger.log_dir).mkdir(parents=True, exist_ok=True)
		kellog.setup_logger(Path(logger.log_dir) / "log.txt")
		kellog.log_args(args, Path(logger.log_dir) / "args.json")

		return logger


# ==================================================================================================
def setup_callbacks(logger: Optional[loggers.TensorBoardLogger], trial: Optional[Trial] = None) -> list:
	callbacks = [ModelSummary(max_depth=-1), RichProgressBar()]
	checkpoint_callback = ModelCheckpoint(
		# monitor=f"loss/{Step.VAL}",
		monitor=f"loss/{Step.TRAIN}",
		auto_insert_metric_name=False,
		filename=f"epoch={{epoch:02d}} train_loss={{loss/{Step.TRAIN}:.2f}}",
		save_top_k=3,
		save_last=True
	)
	lr_callback = LearningRateMonitor(logging_interval="epoch")
	if logger is not None:
		callbacks += [checkpoint_callback, lr_callback]
	if trial is not None:
		# callbacks.append(PyTorchLightningPruningCallback(trial, monitor=f"loss/{Step.VAL}"))
		callbacks.append(PyTorchLightningPruningCallback(trial, monitor=f"loss/{Step.TRAIN}"))

	return callbacks


# ==================================================================================================
def limit_float_int(limit: str) -> Union[float, int]:
	return float(limit) if "." in str(limit) else int(limit)
