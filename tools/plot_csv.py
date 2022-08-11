#!/usr/bin/env python3
from contextlib import suppress
with suppress(ImportError): import colored_traceback.auto
import argparse
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from rich import print, inspect

# ==================================================================================================
class Object():
	# ----------------------------------------------------------------------------------------------
	def __init__(self):
		self.parse_args()
		for csvPath, outDir in zip(self.args.csvPath, self.args.out_dir):
			print(csvPath)
			data = self.read_data(csvPath)
			try:
				fails = self.read_fails(csvPath)
			except IndexError:
				print("Caught error, assuming 'fail reasons' are not present...")
				fails = {}
			try:
				policy_updates = self.read_policy_updates(csvPath)
			except IndexError:
				print("Caught error, assuming 'policy updates' are not present...")
				policy_updates = {}
			self.plot_data(data, fails, policy_updates, outDir)

	# ----------------------------------------------------------------------------------------------
	def parse_args(self):
		parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		parser.add_argument("csvPath", type=Path, help="Input CSV file", nargs="+", metavar="CSV_PATH")
		parser.add_argument("-o", "--out_dir", type=Path, help="Directory", metavar="DIRECTORY")
		parser.add_argument("-s", "--skiplines", type=int, help="Skip N lines in case of comments at the top of the file")
		parser.add_argument("-H", "--head", type=int, default=0, help="Only use N values from the top (after skiplines)")
		parser.add_argument("-t", "--tail", type=int, default=0, help="Only use N values from the bottom")

		self.args = parser.parse_args()
		if not self.args.out_dir:
			self.args.out_dir = [p.parent for p in self.args.csvPath]

	# ----------------------------------------------------------------------------------------------
	def read_data(self, csvPath: Path) -> np.ndarray:
		data = np.genfromtxt(csvPath, delimiter=",", skip_header=self.args.skiplines)
		data = data[:, 0] # Rewards
		data = data[:self.args.head]
		data = data[-self.args.tail:]

		return data

	# ----------------------------------------------------------------------------------------------
	def read_fails(self, csvPath: Path) -> np.ndarray:
		column = 3 # failReason
		with open(csvPath, "r") as file:
			reader = csv.reader(file)
			reasons = {
				"too_far_left": [],
				"too_far_right": [],
				"pole_fell_left": [],
				"pole_fell_right": [],
			}
			for i, line in enumerate(reader):
				if i < self.args.skiplines:
					continue
				if self.args.head > 0 and i >= self.args.head:
					break
				if self.args.tail > 0 and i < self.args.tail:
					continue

				for k in reasons.keys():
					if line[column] == k:
						if len(reasons[k]) == 0:
							reasons[k].append(1)
						else:
							reasons[k].append(reasons[k][-1] + 1)
					else:
						if len(reasons[k]) == 0:
							reasons[k].append(0)
						else:
							reasons[k].append(reasons[k][-1])
				reasons[line[column]]

		for k in reasons.keys():
			reasons[k] = np.array(reasons[k])

		return reasons

	# ----------------------------------------------------------------------------------------------
	def read_policy_updates(self, csvPath: Path):
		column = 4 # policy_updates
		with open(csvPath, "r") as file:
			reader = csv.reader(file)
			updates = []
			for i, line in enumerate(reader):
				if i < self.args.skiplines:
					continue
				if line[column] == "1":
					updates.append(i - self.args.skiplines)

		return updates

	# ----------------------------------------------------------------------------------------------
	def plot_data(self, data: np.ndarray, fails: dict[np.ndarray], policy_updates: np.ndarray, outDir: Path):
		# Seem to need to set zorder in `.plot()` as well as `.set_zorder()`
		# Also seem to need to `.patch.set_visible(False)`
		# Also seem to need to set axes like this
		ax1 = plt.gca()
		ax2 = ax1.twinx()
		ax3 = ax1.twiny()

		# Episode rewards
		ax1.set_ylim([0, 500]) # TODO get env max reward instead of hard coding
		ax1.plot(data, linewidth=1, color="grey", label="reward", zorder=20)
		ax1.plot(self.smooth(data), linewidth=2, color="black", label="reward smoothed", zorder=20)
		l = ax1.legend(loc="center left")
		l.set_zorder(100)

		# Fail reasons
		for failReason, counts in fails.items():
			ax2.plot(counts, linewidth=2, label=failReason, zorder=30)
		if len(fails) > 0:
			l = ax2.legend(loc="upper left")
			l.set_zorder(100)

		# Policy updates
		for update in policy_updates:
			ax3.axvline(x=update, linewidth=1, color="lightgrey", zorder=10)

		ax1.set_zorder(20)
		ax2.set_zorder(30)
		ax3.set_zorder(10)
		ax1.patch.set_visible(False)
		ax2.patch.set_visible(False)
		ax3.patch.set_visible(False)

		# print(max(line.get_zorder() for line in ax1.lines))
		# print(max(line.get_zorder() for line in ax2.lines))
		# print(max(line.get_zorder() for line in ax3.lines))
		# print(f"ax1: {max([_.zorder for _ in ax1.get_children()])}")
		# print(f"ax2: {max([_.zorder for _ in ax2.get_children()])}")
		# print(f"ax3: {max([_.zorder for _ in ax3.get_children()])}")

		plt.savefig(str(outDir / f"plot.png"))
		plt.clf() # Otherwise when doing multiple files, the lines are not cleared

	# ----------------------------------------------------------------------------------------------
	def smooth(self, data, weight: float = 0.95): # Weight between 0 and 1
		last = data[0] # First value in the plot (first timestep)
		smoothed = list()
		for point in data:
			smoothed_val = last * weight + (1 - weight) * point
			smoothed.append(smoothed_val)
			last = smoothed_val

		return smoothed


# ==================================================================================================
def main():
	obj = Object()


# ==================================================================================================
if __name__ == "__main__":
	main()
