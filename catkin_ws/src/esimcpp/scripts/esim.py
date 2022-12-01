#!/usr/bin/env python
import numpy as np
from rich import print, inspect

from esimcpp import SimulatorBridge

# ==================================================================================================
if __name__ == "__main__":
	img = np.random.randint(low=0, high=255, size=(10, 10), dtype=np.uint8)
	# events = img2events(img)

	bridge = SimulatorBridge(
		1.0, # Contrast threshold (positive): double contrast_threshold_pos
		1.0, # Contrast threshold (negative): double contrast_threshold_neg
		0.021, # Standard deviation of contrast threshold (positive): double contrast_threshold_sigma_pos = 0.021
		0.021, # Standard deviation of contrast threshold (negative): double contrast_threshold_sigma_neg = 0.021
		0, # Refractory period (time during which a pixel cannot fire events just after it fired one), in nanoseconds: int64_t refractory_period_ns
		True, # Whether to convert images to log images in the preprocessing step: const bool use_log_image
		0.001, # Epsilon value used to convert images to log: L = log(eps + I / 255.0): const double log_eps
		False, # Whether to simulate color events or not (default: false): const bool simulate_color_events
		# const double exposure_time_ms = 10.0, # Exposure time in milliseconds, used to simulate motion blur
		# const bool anonymous = false, # Whether to set a random number after the /ros_publisher node name (default: false)
		# const int32_t random_seed = 0 # Random seed used to generate the trajectories. If set to 0 the current time(0) is taken as seed.
	)
	events = bridge.img2events(img, 0)

	print(f"len(events): {len(events)}")
	# for event in events:
	# 	print(event)
	# 	# inspect(event)
	# 	# quit(0)
	# 	# print(f"[{event.x} ,{event.y}] {event.t} : {event.pol}")

	img = np.random.randint(low=0, high=255, size=(10, 10), dtype=np.uint8)
	events = bridge.img2events(img, int((1 / 30) * 1e9))
	print(f"len(events): {len(events)}")
