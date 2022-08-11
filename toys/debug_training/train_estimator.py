#!/usr/bin/env python3
import colored_traceback.auto
import argparse
from pathlib import Path
from typing import Optional

from setproctitle import setproctitle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
import wandb
import torchvision
import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from rich import print, inspect

# import rl
from rl.utils.step import Step

# ==================================================================================================
def main():
	setproctitle(Path(__file__).name)
	args = parse_args()

	if args.optuna is not None:
		study = optuna.create_study(
			study_name="RL_estimator",
			direction=optuna.study.StudyDirection.MINIMIZE,
			storage=args.optuna,
			load_if_exists=True,
		)
		optuna.create_study()
		study.optimize(Objective(args), n_trials=100, n_jobs=1, gc_after_trial=False)
		print(f"Best params so far: {study.best_params}")
	else:
		Objective(args)()


# ==================================================================================================
class Objective():
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace):
		self.args = args

	# ----------------------------------------------------------------------------------------------
	def __call__(self, trial: Optional[optuna.trial.Trial] = None):
		# TODO change args to a range instead of hard-coding
		if trial is not None:
			# time.sleep(5)
			self.args.lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
			# --ts
			# --tsamples
			# -b
			# -a
			# -k
			# -c
			# --spatial_decay
		model = Model(self.args, trial)

		if self.args.nolog:
			loggers = []
			callbacks = []
		else:
			logger_tb = TensorBoardLogger(
				save_dir=Path(__file__).parent / "runs",
				name=Path(__file__).stem,
				default_hp_metric=False,
			)
			logger_wandb = WandbLogger(
				save_dir=str(Path(__file__).parent / "runs"),
				name=Path(__file__).stem,
				project="RL_estimator",
				# log_model="all",
			)
			loggers = [logger_tb, logger_wandb]
			callbacks = [
				LearningRateMonitor(),
				ModelCheckpoint(monitor=f"loss/{Step.VAL}", mode="min", save_top_k=3, save_last=True),
			]
			if trial is not None:
				callbacks.append(PyTorchLightningPruningCallback(trial, monitor=f"loss/{Step.VAL}"))
		trainer = pl.Trainer(
			logger=loggers,
			callbacks=callbacks,
			accelerator="cpu" if self.args.cpu else "gpu",
			gpus=0 if self.args.cpu else -1,
			max_epochs=100,
			num_sanity_val_steps=0 if self.args.optuna is not None else 2,
			# enable_model_summary=False, # This is done manually in the callbacks
		)
		trainer.fit(model)

		return trainer.callback_metrics[f"loss/{Step.VAL}"].item()


# ==================================================================================================
class Model(pl.LightningModule):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, args: argparse.Namespace, trial: Optional[optuna.trial.Trial] = None):
		super().__init__()
		self.args = args
		self.save_hyperparameters()
		self.criterion = torch.nn.L1Loss(reduction="none")

		# wandb_logger.watch(model, log="all")

		self.layer1 = torch.nn.Sequential(
			torch.nn.Conv2d(2, 16, 3, stride=2, padding=1),
			torch.nn.ReLU(),
			# torch.nn.Dropout2d(0.1),
			torch.nn.GroupNorm(num_groups=4, num_channels=16),
		)
		self.layer2 = torch.nn.Sequential(
			torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
			torch.nn.ReLU(),
			# torch.nn.Dropout2d(0.1),
			torch.nn.GroupNorm(num_groups=4, num_channels=32),
		)
		self.layer3 = torch.nn.Sequential(
			torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
			torch.nn.ReLU(),
			# torch.nn.Dropout2d(0.1),
			torch.nn.GroupNorm(num_groups=4, num_channels=64),
		)
		self.layer4 = torch.nn.Sequential(
			torch.nn.Flatten(),
			torch.nn.Linear(15360, 256),
			# torch.nn.Dropout(0.5),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 4),
			# torch.nn.Dropout(0.5),
		)
		self.layers = torch.nn.Sequential(
			self.layer1,
			self.layer2,
			self.layer3,
			self.layer4,
		)

	# ----------------------------------------------------------------------------------------------
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

		return [optimizer], {"scheduler": scheduler, "monitor": f"loss/{Step.VAL}"}

	# ----------------------------------------------------------------------------------------------
	def forward(self, x):
		# print(f"x0: {x.shape}")
		# x = self.layer1(x)
		# print(f"x1: {x.shape}")
		# x = self.layer2(x)
		# print(f"x2: {x.shape}")
		# x = self.layer3(x)
		# print(f"x3: {x.shape}")
		# x = self.layer4(x)
		# print(f"x4: {x.shape}")
		x = self.layers(x)

		return x

	# ----------------------------------------------------------------------------------------------
	def step(self, step: Step, batch, batch_idx: int):
		if self.global_step == 0 and not self.args.nolog:
			wandb.define_metric(f"loss/{step}", summary="min")

		gt = batch["gt"]
		obs = batch["obs"]

		pred = self(obs)
		loss = self.criterion(pred, gt)

		loss_ = loss.mean(dim=0)
		self.log(f"loss/{step}_x", loss_[0])
		self.log(f"loss/{step}_xdot", loss_[1])
		self.log(f"loss/{step}_theta", loss_[2])
		self.log(f"loss/{step}_thetadot", loss_[3])
		loss = loss.mean()
		self.log(f"loss/{step}", loss)

		if batch_idx == 0:
			obs = torch.cat((obs, torch.zeros([obs.shape[0], 1, *obs.shape[2:]], device=self.device)), 1)
			images = torchvision.utils.make_grid(obs, nrow=4, pad_value=0.5)
			self.loggers[0].experiment.add_image(f"{step}/obs", images, global_step=self.current_epoch)
			wandb.log({f"{step}/obs": wandb.Image(images, caption=f"Env events input")})
		if step is not Step.TRAIN:
			x_gt, xdot_gt, theta_gt, thetadot_gt = gt[0]
			x_pred, xdot_pred, theta_pred, thetadot_pred = pred[0]
		# 	self.loggers[0].experiment.add_scalar(f"{step}/x_gt", x_gt, global_step=self.current_epoch)
		# 	self.loggers[0].experiment.add_scalar(f"{step}/xdot_gt", xdot_gt, global_step=self.current_epoch)
		# 	self.loggers[0].experiment.add_scalar(f"{step}/theta_gt", theta_gt, global_step=self.current_epoch)
		# 	self.loggers[0].experiment.add_scalar(f"{step}/thetadot_gt", thetadot_gt, global_step=self.current_epoch)
		# 	self.loggers[0].experiment.add_scalar(f"{step}/x_pred", x_pred, global_step=self.current_epoch)
		# 	self.loggers[0].experiment.add_scalar(f"{step}/xdot_pred", xdot_pred, global_step=self.current_epoch)
		# 	self.loggers[0].experiment.add_scalar(f"{step}/theta_pred", theta_pred, global_step=self.current_epoch)
		# 	self.loggers[0].experiment.add_scalar(f"{step}/thetadot_pred", thetadot_pred, global_step=self.current_epoch)
			self.log(f"{step}/x_gt", x_gt)
			self.log(f"{step}/xdot_gt", xdot_gt)
			self.log(f"{step}/theta_gt", theta_gt)
			self.log(f"{step}/thetadot_gt", thetadot_gt)
			self.log(f"{step}/x_pred", x_pred)
			self.log(f"{step}/xdot_pred", xdot_pred)
			self.log(f"{step}/theta_pred", theta_pred)
			self.log(f"{step}/thetadot_pred", thetadot_pred)

		return loss

	# ----------------------------------------------------------------------------------------------
	def training_step(self, batch, batch_idx: int):
		return self.step(Step.TRAIN, batch, batch_idx)

	# ----------------------------------------------------------------------------------------------
	def validation_step(self, batch, batch_idx: int):
		return self.step(Step.VAL, batch, batch_idx)

	# ----------------------------------------------------------------------------------------------
	def dataloader(self, step: Step):
		return DataLoader(
			EnvDataset(self.args.dataset_dir / str(step)),
			batch_size=1 if step is Step.TEST else 8,
			shuffle=step is Step.TRAIN,
			drop_last=step is Step.TRAIN,
			num_workers=20,
			pin_memory=True,
		)

	# ----------------------------------------------------------------------------------------------
	def train_dataloader(self):
		return self.dataloader(Step.TRAIN)

	# ----------------------------------------------------------------------------------------------
	def val_dataloader(self):
		return self.dataloader(Step.VAL)


# ==================================================================================================
class EnvDataset(Dataset):
	# ----------------------------------------------------------------------------------------------
	def __init__(self, root_dir: Path):
		self.paths = list(root_dir.glob("*.npz"))

	# ----------------------------------------------------------------------------------------------
	def __len__(self):
		return len(self.paths)

	# ----------------------------------------------------------------------------------------------
	def __getitem__(self, idx):
		data = np.load(self.paths[idx], allow_pickle=True)
		obs = data["obs"]
		gt = data["gt"]

		obs = torch.tensor(obs)
		gt = torch.tensor(gt)

		obs = obs.sum(1)

		sample = {
			"gt": gt,
			"obs": obs,
		}

		return sample


# ==================================================================================================
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-d", "--dataset_dir", type=Path, default=Path(__file__).parent / "dataset", help="Dataset directory")
	parser.add_argument("-n", "--nolog", action="store_true", help="Suppress logging")
	parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--optuna", type=str, help="Optimise with optuna using this storage URL. Examples: 'sqlite:///optuna.db' or 'postgresql://postgres:password@host:5432/postgres'")

	return parser.parse_args()


# ==================================================================================================
if __name__ == "__main__":
	main()
