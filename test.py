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
	Model = getattr(rl.models, args.model)
	model = Model(args, trial=None)
	trainer = pl.Trainer(
		# logger=logger,
		gpus=0 if args.cpu else -1,
	)
	path = utils.get_checkpoint(logger, args.checkpoint)
	if path is not None and path.exists():
		model = Model.load_from_checkpoint(path)
		trainer.test(model=model) # ckpt_path doesn't get hyperparameters/args from the checkpoint


# ==================================================================================================
if __name__ == "__main__":
	torch.set_printoptions(precision=16, sci_mode=False)
	args = utils.parse_args()
	if args.seed is not None:
		pl.utilities.seed.seed_everything(seed=args.seed, workers=True)
	main(args)
