'''
allows viewing images from the dataset
'''
import os
import subprocess
import tempfile
from typing import Iterable, List, Union
import cv2
import numpy as np
# from dataio.dataset import CochlearIqDataset
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
	dtype: Union[type, str] = 'float32'

	def __init__(
		self,
		dicom_paths: List[str],
		labels: List[str] = None,  # separate folders by labels
		img_names: List[str] = None,  # name images, by default images are named img_i.png
		buffers: List[str] = None,  # numpy buffers
		):
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
	
	# @classmethod
	# def from_dataset(dataset: CochlearIqDataset):
	# 	return DataViewer(dataset.dicom_paths, dataset.labels, dataset.buffer_paths)

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
		dir_path = os.path.join(self.tempdir, str(label))
		full_img_path = os.path.join(dir_path, img_name
									+ ('.png' if not img_name.endswith('.png') else ''))
		cv2.imwrite(full_img_path, 255*img_array)

	def run(self, verbose: bool = True):
		# make temp directory
		if self.tempdir is None:
			self.tempdir = tempfile.mkdtemp()
		# create subdirectories by labels
		for label in np.unique(self.labels):
			os.mkdir(os.path.join(self.tempdir, str(label)))
		# write img_array into corresponding path
		for i, dicom_path in enumerate(self.dicom_paths):
			if self.labels is not None:
				label = self.labels[i]
			else:
				label = None
			if self.buffers is not None:
				try:
					img_array = self.buffers[i]
				except FileNotFoundError():
					img_array = self._read_image(dicom_path)
			else:
				img_array = self._read_image(dicom_path)
			if self.img_names is not None:
				img_name = self.img_names[i]
			else:
				img_name = 'img' + str(i)
			self._write_image(img_array, img_name, label)
		# show path of the temp directory
		if verbose:
			print(f'{i} images saved to {self.tempdir}')

	def open(self):
		# open self.tempdir, 
		subprocess.Popen(self.tempdir)
