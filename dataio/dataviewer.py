'''
allows viewing images from the dataset
'''
import os
import subprocess
import tempfile
from typing import Iterable, List, Union
import cv2
import numpy as np
import torch
from dataio.dataset import CochlearIqDataset
from dataio.utility import read_dicom_image, normalize_image


class DataViewer:
	'''
	class to view data from the 
	'''
	tempdir: os.PathLike = None
	dicom_paths: Iterable[str]
	labels: Iterable[str]
	buffers: Iterable[str] = None
	img_names: Iterable[str] = None
	_unique_labels: Iterable[str] = None
	_from_dataset: bool = False
	dataset: CochlearIqDataset = None

	def __init__(
		self,
		dicom_paths: List[str] = None,
		labels: List[str] = None,  # separate folders by labels
		img_names: List[str] = None,  # name images, by default images are named img_i.png
		buffers: List[str] = None,  # numpy buffers
		dataset: CochlearIqDataset = None,
		):
		'''
		inputs:
			option 1:
				dicom_paths: 
			option 2:
				dataset: the dataset class, use the image arrays and labels from the dataset
		'''
		if dataset is None:
			# option 1: initiating with dicom_path, labels, img_names and buffers
			self.dicom_paths = dicom_paths
			if labels is not None:
				assert len(labels) == len(dicom_paths)
				self.labels = labels
			if img_names is not None:
				assert len(img_names) == len(dicom_paths)
				self.img_names = img_names
			if buffers is not None:
				assert len(buffers) == len(dicom_paths)
				self.buffers = buffers
		else:
			# option 2: initiating with dataset
			self.dataset = dataset
		self.run()
	
	# @classmethod
	# def from_dataset(dataset: CochlearIqDataset):
	# 	return DataViewer(dataset.dicom_paths, dataset.labels, dataset.buffer_paths)
	@classmethod
	def from_dataset(cls, dataset: CochlearIqDataset):
		data_viewer = cls(dataset=dataset)
		return data_viewer

	def _read_image(self, dicom_path: str):
		img_array = read_dicom_image(dicom_path)
		norm_img_array = normalize_image(img_array)
		return norm_img_array

	def _read_numpy(self, numpy_path: str):
		if not os.path.isfile(numpy_path):
			raise FileNotFoundError('buffer file {numpy_path} not found')
		img_array = np.load(numpy_path)
		return img_array

	def _write_image(self, img_array: np.ndarray, img_name: str, label=None):
		if self.tempdir is None:
			raise FileNotFoundError('temp folder not found')
		# if label is none, put under sorting directory
		if label is not None:
			dir_path = os.path.join(self.tempdir, str(label))
		else:
			dir_path = self.tempdir
		# combine with img_name to get the full path
		full_img_path = os.path.join(dir_path, img_name
									+ ('.png' if not img_name.endswith('.png') else ''))
		# write image
		cv2.imwrite(full_img_path, 255*img_array)

	# iterate items (images, labels), by reading the dicoms
	def _iter_from_attributes(self):
		'''
		iteratively return images arrays and labels
		using dicom paths and labels in class attributes
		'''
		for i, dicom_path in enumerate(self.dicom_paths):
			if self.labels is not None:
				label = self.labels[i]
			else:
				label = None
			# read image from buffers, otherwise 
			if self.buffers is not None:
				buffer_file = self.buffers[i]
				try:
					assert os.path.isfile(buffer_file)
					img_array = np.load(self.buffers[i])
				except AssertionError():
					img_array = self._read_image(dicom_path)
			else:
				img_array = self._read_image(dicom_path)
			if self.img_names is not None:
				img_name = self.img_names[i]
			else:
				img_name = 'img' + str(i)
			yield img_array, img_name, label
	
	def _iter_from_dataset(self):
		'''
		iteratively return image arrays and labels
		using dataset __getitem__ method
		'''
		for i, (image_tensor, label_tensor) in enumerate(self.dataset):
			img_array = image_tensor[0].numpy()
			img_name = 'img_' + str(i)
			label = torch.argmax(label_tensor).item() + 1
			yield img_array, img_name, label

	def _iter(self):
		'''
		return a generator that generates img_array, img_name, label
		'''
		if self.dataset is None:
			return self._iter_from_attributes()
		else:
			return self._iter_from_dataset()

	# iterate items with dataset 

	def run(self, verbose: bool = True):
		'''
		write images into the temporary paths
		'''
		# make temp directory
		if self.tempdir is None:
			self.tempdir = tempfile.mkdtemp(prefix='ciiq_dataviewer_')
		# iterate through self._iter() method
		for img_array, img_name, label in self._iter():
			# create label subdirectory if not exists
			label_dir = os.path.join(self.tempdir, str(label))
			if not os.path.isdir(label_dir):
				os.mkdir(label_dir)
			# write image to label subdirectory
			self._write_image(img_array, img_name, label)
		if verbose:
			print(f'images saved to {self.tempdir}')

		# ### CUTOVER: use new iterators for image array and labels

		# # make temp directory
		# if self.tempdir is None:
		# 	self.tempdir = tempfile.mkdtemp()
		# # create subdirectories by labels
		# for label in np.unique(self.labels):
		# 	os.mkdir(os.path.join(self.tempdir, str(label)))
		# # write img_array into corresponding path
		# for i, dicom_path in enumerate(self.dicom_paths):
		# 	if self.labels is not None:
		# 		label = self.labels[i]
		# 	else:
		# 		label = None
		# 	# read img_array from buffers, or from dicom if fails
		# 	if self.buffers is not None:
		# 		try:
		# 			img_array = self.buffers[i]
		# 		except FileNotFoundError():
		# 			img_array = self._read_image(dicom_path)
		# 	else:
		# 		img_array = self._read_image(dicom_path)
		# 	# assign image names if given
		# 	if self.img_names is not None:
		# 		img_name = self.img_names[i]
		# 	else:
		# 		img_name = 'img' + str(i)
		# 	# apply transforms if given
		# 	# write image array to file
		# 	self._write_image(img_array, img_name, label)
		# # show path of the temp directory
		# if verbose:
		# 	print(f'{i} images saved to {self.tempdir}')

	def open(self):
		# open self.tempdir, 
		subprocess.Popen(self.tempdir)
