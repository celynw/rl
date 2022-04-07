#!/usr/bin/env python3
import argparse
from contextlib import suppress
with suppress(ImportError): import colored_traceback.auto

import torch
import pytorch_lightning as pl
from kellog import info, warning, error, debug
from rich import print, inspect

import rl
from rl import utils

# ==================================================================================================
def main(args: argparse.Namespace):
	logger = utils.setup_logger(args, suppress_output=True)
	checkpointPath = utils.get_checkpoint(logger, args.checkpoint)
	Model = getattr(rl.models, args.model)
	model = Model.load_from_checkpoint(checkpointPath)
	model.hparams.epoch_length = args.epoch_length
	trainer = pl.Trainer(
		logger=logger,
		gpus=0 if args.cpu else -1,
		callbacks=utils.setup_callbacks(None, Model.monitor, Model.monitor_dir),
		max_epochs=args.max_epochs,
	)
	if checkpointPath is not None and checkpointPath.exists():
		trainer.test(model, ckpt_path=checkpointPath) # ckpt_path doesn't get hyperparameters/args from the checkpoint


# ==================================================================================================
if __name__ == "__main__":
	torch.set_printoptions(precision=16, sci_mode=False)
	args = utils.parse_args()
	pl.utilities.seed.seed_everything(seed=args.seed if args.seed is not None else 0, workers=True)
	main(args)
