import torch
import msgpack

# ==================================================================================================
def generate_events(txn, generator, valid_stamps):
	events = []
	for stamp in valid_stamps:
		ev = msgpack.loads(txn.get(key=msgpack.dumps(stamp)))
		stamps = ev[0]
		polarities = ev[1]
		xs = ev[2]
		ys = ev[3]
		for stamp, polarity, x, y in zip(stamps, polarities, xs, ys):
			events.append([polarity, x, y, stamp])
	polarities = torch.tensor([e[0] for e in events])
	coords = torch.tensor([[e[1], e[2]] for e in events])
	stamps = torch.tensor([e[3] for e in events], dtype=torch.double)
	assert(polarities.shape[0] == coords.shape[0] == stamps.shape[0])

	events = generator.getSlayerSpikeTensor(polarities, coords, stamps)
	events = events.permute(0, 3, 1, 2) # CHWT -> CDHW

	return events
