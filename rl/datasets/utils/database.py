#!/usr/bin/env python3
from pathlib import Path
import io
import tempfile
from typing import Union

import lmdb
import msgpack
from PIL import Image, ImageFile
import numpy as np
import torch
from kellog import info, warning, error, debug
from rich import print, inspect

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==================================================================================================
class Database():
	_database = None
	_protocol = None
	_length = None
	# ----------------------------------------------------------------------------------------------
	def __init__(self, path: Path, readahead: bool = False, pre_open: bool = False):
		"""
		Base class for LMDB-backed databases.

		Args:
			path (Path): Path to the database.
			readahead (bool, optional): Enables the filesystem readahead mechanism. Defaults to False.
			pre_open (bool, optional): If set to True, the first iterations will be faster, but it will raise error when doing multi-gpu training. If set to False, the database will open when you will retrieve the first item. Defaults to False.

		Raises:
			FileNotFoundError: [description]
		"""
		self.path = path
		self.readahead = readahead
		self.pre_open = pre_open

		if not self.path.exists():
			raise FileNotFoundError(self.path)
		self._has_fetched_an_item = False

	# ----------------------------------------------------------------------------------------------
	@property
	def database(self):
		if self._database is None:
			self._database = lmdb.open(
				path=self.path,
				readonly=True,
				readahead=self.readahead,
				max_spare_txns=256,
				lock=False,
			)

		return self._database

	# ----------------------------------------------------------------------------------------------
	@database.deleter
	def database(self):
		if self._database is not None:
			self._database.close()
			self._database = None

	# ----------------------------------------------------------------------------------------------
	@property
	def protocol(self) -> set:
		"""
		Read the msgpack protocol contained in the database.

		Returns:
			set: The set of available keys.
		"""
		if self._protocol is None:
			self._protocol = self._get(
				item="protocol",
				convert_key=lambda key: key.encode("ascii"),
				convert_value=lambda value: msgpack.loads(value),
			)
		return self._protocol

	# ----------------------------------------------------------------------------------------------
	@property
	def keys(self) -> set:
		"""
		Read the keys contained in the database.

		Returns:
			set: The set of available keys.
		"""
		protocol = self.protocol
		keys = self._get(
			item="keys",
			convert_key=lambda key: msgpack.dumps(key, protocol=protocol),
			convert_value=lambda value: msgpack.loads(value),
		)

		return keys

	# ----------------------------------------------------------------------------------------------
	def __len__(self) -> int:
		"""
		Returns the number of keys available in the database.

		Returns:
			int: The number of keys.
		"""
		if self._length is None:
			self._length = len(self.keys)

		return self._length

	# ----------------------------------------------------------------------------------------------
	def __getitem__(self, item):
		"""
		Retrieves an item or a list of items from the database.

		Args:
			item ([type]): A key or a list of keys.

		Returns:
			[type]: A value or a list of values.

		"""
		self._has_fetched_an_item = True
		if not isinstance(item, list):
			item = self._get(item, self._convert_key, self._convert_value)
		else:
			item = self._gets(item, self._convert_keys, self._convert_values)

		return item

	# ----------------------------------------------------------------------------------------------
	def _get(self, item, convert_key, convert_value):
		"""
		Instantiates a transaction and its associated cursor to fetch an item.

		Args:
			item: A key.
			convert_key:
			convert_value:

		Returns:
			[type]:
		"""
		with self.database.begin() as txn:
			with txn.cursor() as cursor:
				item = self._fetch(cursor, item, convert_key, convert_value)
		self._keep_database()

		return item

	# ----------------------------------------------------------------------------------------------
	def _gets(self, items, convert_keys, convert_values):
		"""
		Instantiates a transaction and its associated cursor to fetch a list of items.

		Args:
			items: A list of keys.
			convert_keys:
			convert_values:

		Returns:
			[type]:
		"""
		with self.database.begin() as txn:
			with txn.cursor() as cursor:
				items = self._fetchs(cursor, items, convert_keys, convert_values)
		self._keep_database()

		return items

	# ----------------------------------------------------------------------------------------------
	def _fetch(self, cursor, key, convert_key, convert_value):
		"""
		Retrieve a value given a key.

		Args:
			cursor:
			key: A key.
			convert_key:
			convert_value:

		Returns:
			[type]: A value.
		"""
		key = convert_key(key=key)
		value = cursor.get(key=key)
		value = convert_value(value=value)

		return value

	# ----------------------------------------------------------------------------------------------
	def _fetchs(self, cursor, keys, convert_keys, convert_values):
		"""
		Retrieve a list of values given a list of keys.

		Args:
			cursor:
			keys: A list of keys.
			convert_keys:
			convert_values:

		Returns:
			[type]: A list of values.
		"""
		keys = convert_keys(keys=keys)
		_, values = list(zip(*cursor.getmulti(keys)))
		values = convert_values(values=values)

		return values

	# ----------------------------------------------------------------------------------------------
	def _convert_key(self, key):
		"""
		Converts a key into a byte key.

		Args:
			key: A key.

		Returns:
			[type]: A byte key.
		"""
		return msgpack.dumps(key, protocol=self.protocol)

	# ----------------------------------------------------------------------------------------------
	def _convert_keys(self, keys: list):
		"""
		Converts keys into byte keys.

		Args:
			keys (list): A list of keys.

		Returns:
			[type]: A list of byte keys.
		"""
		return [self._convert_key(key=key) for key in keys]

	# ----------------------------------------------------------------------------------------------
	def _convert_value(self, value):
		"""
		Converts a byte value back into a value.

		Args:
			value: A byte value.

		Returns:
			[type]: A value
		"""
		return msgpack.loads(value)

	# ----------------------------------------------------------------------------------------------
	def _convert_values(self, values):
		"""
		Converts bytes values back into values.

		Args:
			values: A list of byte values.

		Returns:
			[type]: A list of values.
		"""
		return [self._convert_value(value=value) for value in values]

	# ----------------------------------------------------------------------------------------------
	def _keep_database(self):
		"""
		Checks if the database must be deleted.

		Returns:
			[type]:
		"""
		if not self.pre_open and not self._has_fetched_an_item:
			del self.database

	# ----------------------------------------------------------------------------------------------
	def __iter__(self):
		"""
		Provides an iterator over the keys when iterating over the database.

		Returns:
			[type]: An iterator on the keys.
		"""
		return iter(self.keys)

	# ----------------------------------------------------------------------------------------------
	def __del__(self):
		"""Closes the database properly."""
		del self.database


# ==================================================================================================
class ImageDatabase(Database):
	# ----------------------------------------------------------------------------------------------
	def _convert_value(self, value):
		"""
		Converts a byte image back into a PIL Image.

		Args:
			value: A byte image.

		Returns:
			[type]: A PIL Image image.
		"""
		return Image.open(io.BytesIO(value))


# ==================================================================================================
class MaskDatabase(ImageDatabase):
	# ----------------------------------------------------------------------------------------------
	def _convert_value(self, value):
		"""
		Converts a byte image back into a PIL Image.

		Args:
			value: A byte image.

		Returns:
			[type]: A PIL Image image.
		"""
		return Image.open(io.BytesIO(value)).convert("1")


# ==================================================================================================
class LabelDatabase(Database):
	pass


# ==================================================================================================
class ArrayDatabase(Database):
	_dtype = None
	_shape = None
	# ----------------------------------------------------------------------------------------------
	@property
	def dtype(self):
		if self._dtype is None:
			protocol = self.protocol
			self._dtype = self._get(
				item="dtype",
				convert_key=lambda key: msgpack.dumps(key, protocol=protocol),
				convert_value=lambda value: msgpack.loads(value),
			)
		return self._dtype

	# ----------------------------------------------------------------------------------------------
	@property
	def shape(self):
		if self._shape is None:
			protocol = self.protocol
			self._shape = self._get(
				item="shape",
				convert_key=lambda key: msgpack.dumps(key, protocol=protocol),
				convert_value=lambda value: msgpack.loads(value),
			)
		return self._shape

	def _convert_value(self, value):
		return np.frombuffer(value, dtype=self.dtype).reshape(self.shape)

	def _convert_values(self, values):
		return np.frombuffer(b"".join(values), dtype=self.dtype).reshape((len(values),) + self.shape)


# ==================================================================================================
class TensorDatabase(ArrayDatabase):
	# ----------------------------------------------------------------------------------------------
	def _convert_value(self, value):
		return torch.from_numpy(super()._convert_value(value))

	# ----------------------------------------------------------------------------------------------
	def _convert_values(self, values):
		return torch.from_numpy(super()._convert_values(values))


# # ==================================================================================================
# class PointcloudDatabase(Database):
# 	# ----------------------------------------------------------------------------------------------
# 	def __init__(self, path: Path):
# 		raise NotImplementedError("Use ArrayDatabase and numpy instead...")
# 		super().__init__(path)

# 	# ----------------------------------------------------------------------------------------------
# 	def __getitem__(self, item):
# 		key = msgpack.dumps(str(item))
# 		with self.db.begin() as txn:
# 			# o3d.io can't read from open files, grr.
# 			with tempfile.NamedTemporaryFile(suffix=".pcd") as f:
# 			# with tempfile.SpooledTemporaryFile() as f:
# 				f.write(txn.get(key))
# 				f.seek(0)
# 				pcl = o3d.io.read_point_cloud(f.name)

# 		return np.asarray(pcl.points)
