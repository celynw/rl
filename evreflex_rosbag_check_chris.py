#!/usr/bin/env python
# -*- coding: future_fstrings -*-
import yaml
import cv2
from cv_bridge import CvBridge
import numpy as np
from rosbag import Bag
from tqdm import tqdm
from pathlib2 import Path

OBJECT = 40
HEIGHT = 260
WIDTH = 346

bridge = CvBridge() # We don't need to re-instantiate this every loop! Only once

# Get list of only bag files in current dir
# rglob does a recursive (r) search using globs (the asterisk)
# pathlib is built-in now, but python 2 is just quite old
# I would really recommend using it instead of `os` as much as you possibly can
bagFiles = Path(".").rglob("*.bag")

# Avoid manual counters if you can
# But if you're using enumerate, you don't need to call your variable `i`
for i, bagFile in enumerate(bagFiles):
	# f-strings are amazing.
	# Again, they're built-in to python3, but not python 2
	# You can access them though by using the coding: future_fstrings line at the top
	print(f"Reading bag {i + 1} of {len(bagFiles)}: {bagFile}")
	# By opening the rosbag like this, it will close itself when we go out of scope. Rest of code touching this is unchanged
	with Bag(bagFile.resolve(), "r") as bag:
		bagName = bag.filename # This is actually also already a pathlib Path object!

		# This is a much cleaner way to creating directories using pathlib
		dataDir = Path("evreflex_data")
		dataDir.mkdir(exist_ok=True, parents=True)
		imgDir = Path("evreflex_imgs")
		imgDir.mkdir(exist_ok=True, parents=True)

		# But pathlib won't copy files, you would best use shutil for that
		# shutil.copyfile(bagName, folder + '/' + bagName)

		# This just checks that the topics we want are in the bag, and tells us how many messages to expect
		# It's not strictly necessary
		wantedTopics = ["/cam0/events", "/cam0/image_alpha"]
		bagInfo = yaml.load(bag._get_yaml_info(), Loader=yaml.FullLoader)
		# bagInfo = yaml.load(bag._get_yaml_info())
		topics = [t["topic"] for t in bagInfo["topics"] if t["topic"] in wantedTopics]
		assert wantedTopics == topics # Quites with error if the topics we wan't aren't in the bag

		# We'll update this EVERY event message we get
		# And we'll reset it every time we write the image out
		img_array = np.zeros((HEIGHT, WIDTH, 3), np.float32)

		# This is ONLY to give us a total number of messages for our pretty tqdm progressbar. Otherwise we don't care
		total = sum([topic["messages"] for topic in bagInfo["topics"] if topic["topic"] in topics])
		# We also don't need the `bar` variable, we can load it directly like this
		# A simpler version would be:
		# 	for i, (topic, msg, t) in enumerate(bag.read_messages(topics=topics)):
		# Or even:
		# 	for i, (topic, msg, t) in enumerate(bag.read_messages()):
		# Or even:
		# 	for topic, msg, t in bag.read_messages():
		for i, (topic, msg, t) in enumerate(tqdm(bag.read_messages(topics=topics), total=total)):
			# With pathlib, the divide operator actually functions as a flexible path separator!
			# Well done for figuring out how to do array slicing like this [:-4] to cut strings
			# But since bagName is a pathlib Path object, we can reliably get the name without the extension just by using `.stem`
			filename = dataDir / f"{bagName.stem}-{i}.txt"
			filename_png = imgDir / f"{bagName.stem}-{i}.png"
			if topic == "/cam0/image_alpha":
				seg = np.asarray(bridge.imgmsg_to_cv2(msg, msg.encoding))
				# seg = seg * np.in1d(seg, labels).reshape(seg.shape) # Remove non-class values
				seg[seg != OBJECT] = 0
				seg[seg == OBJECT] = 255

				kernel = np.ones((5, 5), np.uint8)
				seg = cv2.erode(seg, kernel)
				seg = cv2.dilate(seg, kernel)

				max_x = 0
				max_y = 0
				min_x = 346
				min_y = 260

				# Don't call this `n_events`
				# It's not events, it's pixels, not actually to do with events, so don't confuse yourself with the variable name
				n_pixels = 0

				# There's definitely better ways to do this with numpy, but I'll leave it alone for you here
				for x in range(0, 346):
					for y in range(0, 260):
						if seg[y][x] == 255:
							# You can do `+=` instead, it's more compact
							# But you can't do `++` in python
							n_pixels += 1

							if max_x < x:
								max_x = x
							if max_y < y:
								max_y = y
							if min_x > x:
								min_x = x
							if min_y > y:
								min_y = y

				if n_pixels > 7:
					# The brackets around the `(float)` didn't do anything in python
					cen_x = float(max_x + min_x) / 2
					cen_x = cen_x / 346
					cen_y = float(max_y + min_y) / 2
					cen_y = cen_y / 260
					height = float(max_y - min_y) / 346
					width = float(max_y - min_y) / 260

					# It's better practice to wrap the open in a with structure, which closes it automatically when it goes out of scope
					with open(filename, "w") as file:
						# Maybe you can see that f-strings are a bit like `.format`
						# file.write("0 {0:.6f} {0:.6f} {0:.6f} {0:.6f}".format(cen_x, cen_y, width, height))
						file.write(f"0 {cen_x:.6f} {cen_y:.6f} {width:.6f} {height:.6f}")
				# BUT, cv2 is stuck in the past, and you'll need to cast the pathlib path to a string for it to work in this particular function
				cv2.imwrite(str(filename_png), img_array)
				img_array = np.zeros((HEIGHT, WIDTH, 3), np.float32) # Reset the event image to start accumulating again
			if topic == "/cam0/events":
				for event in msg.events:
					if event.polarity == True:
						img_array[event.y, event.x, 1] = 255
					else:
						img_array[event.y, event.x, 2] = 255
				# Don't write the image out here, we'll do it each time we get a segmentation image
		# bag.close # We don't need to do this, it will happen automatically when the line:
		# 	with Bag(bagFile.resolve(), "r") as bag:
		# goes out of scope
	# quit(0) # We also don't need to do this
