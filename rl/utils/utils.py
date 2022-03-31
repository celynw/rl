#!/usr/bin/env python3
import argparse
from pathlib import Path
import inspect
import time
from typing import Union, Optional

import torch
from git import Repo
import setproctitle
from kellog import info, warning, error, debug
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import RichModelSummary, LearningRateMonitor, RichProgressBar
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from optuna.trial import Trial
import kellog
from kellog import info, warning, error, debug
from rich import inspect as rinspect

import rl
from rl.utils import ArgumentParser, Step, ModelCheckpointBest

eps = 1e-15
# eps = torch.finfo(tensor.dtype).eps

# ==================================================================================================
def parse_args() -> argparse.Namespace:
	# Help argument must only be added once, so we go through as many arg groups as we can first
	parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)

	group = parser.add_argument_group("Trainer")
	models = [name for name, obj in inspect.getmembers(rl.models) if inspect.isclass(obj) and name != "Base"]
	# TODO also exclude other parent classes
	group.add_argument("model", choices=models, metavar=f"MODEL: {{{', '.join(models)}}}", help="Model to train")
	args_known, _ = parser.parse_known_args()
	if args_known.model is None:
		parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")

	group.add_argument("-o", "--output_directory", type=Path, help="Output directory")
	group.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
	group.add_argument("-v", "--version", type=str, help="Try to continue training from this version", default=None)
	group.add_argument("--cpu", action="store_true", help="Use CPU rather than GPU as device")
	group.add_argument("--optuna", type=str, help="Optimise with optuna using this storage URL. Examples: 'sqlite:///optuna.db' or 'postgresql://postgres:password@host:5432/postgres'")
	group.add_argument("--profile", action="store_true", help="Profile a single epoch with tensorboard")
	group.add_argument("--no_graph", action="store_true", help="Disable tensorboard graph logging (sometimes it just won't play nice)")
	group.add_argument("--seed", type=int, help="Use specified random seed for everything", default=None)
	group.add_argument("--train_lim", type=str, help="Use this train proportion (float) or batches (int) each epoch (still randomised over entire dataset)", default="1.0")
	group.add_argument("--val_lim", type=str, help="Use this val proportion (float) or batches (int) each epoch (still randomised over entire dataset)", default="1.0")
	group.add_argument("--test_lim", type=str, help="Use this test proportion (float) or batches (int) each epoch (still randomised over entire dataset)", default="1.0")
	group.add_argument("--overfit", type=str, help="Overfit to this proportion (float) or batches (int), use train set for val", default="0.0")
	group.add_argument("-C", "--checkpoint", type=Path, help="Path to checkpoint file. Can be relative to specific model log directory", default="last.ckpt")
	group.add_argument("-A", "--autoLR", action="store_true", help="First run the auto learning rate finder")
	import __main__
	group.add_argument("--proctitle", type=str, help="Process title", default=Path(__main__.__file__).name)

	Model = getattr(rl.models, args_known.model)
	parser = Model.add_argparse_args(parser) # Will chain to all child parser groups

	try: # If we didn't add the help already (because an essential option was missing), add it now
		parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")
	except argparse.ArgumentError:
		pass

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
def get_checkpoint(logger: loggers.TensorBoardLogger, checkpoint: Union[str, Path]) -> Optional[Path]:
	if logger is None:
		checkpointPath = None
	else:
		ckptDir = Path(logger.log_dir) / "checkpoints"
		checkpointPath = ckptDir / checkpoint
		if not checkpointPath.exists():
			checkpointPath = checkpoint
		if not checkpointPath.exists():
			if checkpoint is not None:
				warning(f"Specified checkpoint not found at '{checkpointPath}'!")
			checkpointPath = None

		return checkpointPath


# ==================================================================================================
def setup_logger(args) -> Optional[loggers.TensorBoardLogger]:
	if args.output_directory is None:
		warning("No output directory specified, will not log!")
		return None
	else:
		logger = loggers.TensorBoardLogger(
			name=args.model,
			save_dir=args.output_directory,
			# sub_dir=getattr(rl.models, args.model).datasetType.__name__,
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
def setup_callbacks(logger: Optional[loggers.TensorBoardLogger], monitor: str, trial: Optional[Trial] = None) -> list:
	callbacks = [RichProgressBar()]
	if trial is None:
		callbacks += [RichModelSummary(max_depth=-1)]
	checkpoint_callback = ModelCheckpointBest(
		monitor=monitor,
		# auto_insert_metric_name=False,
		# filename=f"epoch={{epoch:02d}} {monitor}={{{monitor}:.2f}}",
		save_top_k=3,
		save_last=True
	)
	lr_callback = LearningRateMonitor(logging_interval="epoch")
	if logger is not None:
		callbacks += [checkpoint_callback, lr_callback]
	if trial is not None:
		callbacks.append(PyTorchLightningPruningCallback(trial, monitor=monitor))

	return callbacks


# ==================================================================================================
def limit_float_int(limit: str) -> Union[float, int]:
	return float(limit) if "." in str(limit) else int(limit)
