#!/usr/bin/env python
import colored_traceback.auto
import argparse
from pathlib import Path

try:
	from cv_bridge import CvBridge
except ModuleNotFoundError as e:
	error(e)
	error("Probably can't find ROS, did you unset PYTHONPATH?")
	quit(1)
from tqdm import tqdm
from rosbag import Bag
import yaml
import numpy as np
from PIL import Image
from rich import print, inspect

# ==================================================================================================
def main(args):
	bridge = CvBridge()
	labels = [i + 1 for i in range(40)] # Match the values in model conversion script...

	with Bag(str(args.rosbag.resolve()), "r") as bag:
		bagInfo = yaml.load(bag._get_yaml_info(), Loader=yaml.FullLoader)
		topics = [t["topic"] for t in bagInfo["topics"] if t["topic"] == "/cam0/image_alpha"]

		seg = None
		total = sum([topic["messages"] for topic in bagInfo["topics"] if topic["topic"] == "/cam0/image_alpha"])
		bar = tqdm(bag.read_messages(topics=topics), total=total)
		for i, (topic, msg, t) in enumerate(bar):
			if topic == "/cam0/image_alpha":
				seg = np.asarray(bridge.imgmsg_to_cv2(msg, msg.encoding))
				# seg = seg * np.in1d(seg, labels).reshape(seg.shape) # Remove non-class values
				if i == 74:
					print(msg.header)
					print(msg.encoding)
					print(seg.shape)
					# unique = np.unique(seg, return_counts=True)
					# unique = zip(list(unique[0]), list(unique[1]))
					# print(list(unique))
					print(seg.dtype)
					seg[seg != 40] = 0
					seg[seg == 40] = 255

					img = Image.fromarray(seg)
					img.show()
					quit(0)


# ==================================================================================================
def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("rosbag", type=Path, help="Path to the input rosbag")

	return parser.parse_args()


# ==================================================================================================
if __name__ == "__main__":
	main(parse_args())
