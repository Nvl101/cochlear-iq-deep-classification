'''
function to split training, validation and test sets
'''
import os
from typing import Iterable, Union
import random
import cv2
import numpy as np
from pydicom import dcmread
import torch

# Methods to normalize images and labels
# method to read image from dicom file
def read_dicom_image(
    dicom_path: str,
    dtype: Union[str, type] = None
    ) -> np.ndarray:
    '''
    inputs:
        `dicom_path`: path of the dicom file
        `dtype`: data type of the output array
    output:
        `img_array`: np.ndarray
    '''
    if not os.path.isfile(dicom_path):
        raise FileNotFoundError(f'dicom file not found: {dicom_path}')
    ds = dcmread(dicom_path)
    pixel_array = ds.pixel_array
    if dtype is not None:
        pixel_array = pixel_array.astype(dtype)
    return pixel_array

# write image arrays into png file
def write_image(
    img_array: np.ndarray,
    img_path: str,
    ):
    '''
    write img array into path
    inputs:
        img_array: greyscale image array of float from 0 to 1
        img_path: target path of image file
    '''
    if os.path.isfile(img_path):
        raise FileExistsError(f'{img_path} already exists')
    cv2.imwrite(img_path, 255 * img_array)

# methods dataset processing
def normalize_image_hu(
    image_array: np.ndarray,
    ):
    '''
    normalize image array using Hounsfield units (HU)

    from -1024, 3071 to range of 0 to 1
    '''
    upper_bound = 3071 if upper_bound is None else upper_bound
    lower_bound = -1024 if lower_bound is None else lower_bound
    img_delta = upper_bound - lower_bound
    norm_image = image_array + lower_bound
    norm_image = norm_image / img_delta
    return norm_image

def normalize_image(
    image_array: np.ndarray,
    datatype: str = 'float32'
    ):
    '''
    normalize image array using maximum and minimum values
    but still capped with lower and upper bounds according to hounsfield

    input:
        image_array, integers ranging from 0 to over 4k

    output:
        norm_image, ranging from 0.0 to 1.0
    '''
    upper_bound = max(np.max(image_array), 3071) # capping upper bound to above 3071
    lower_bound = min(np.min(image_array), 0) # lower bound no more than 0
    img_delta = upper_bound - lower_bound
    norm_image = image_array + lower_bound
    norm_image = norm_image / img_delta
    norm_image = norm_image.astype(datatype)
    return norm_image

def normalize_label(
    label: int,
    n_classes: int = 3,
    ):
    '''
    normalize the labels to binary
    e.g. in a 3 class label,
    2 -> [0, 0, 1]
    
    input:
        label: class number of the label
        n_classes: total number of classes
    output:
        norm_label: 
    '''
    label_no = int(label)
    norm_label = torch.zeros(n_classes, dtype=torch.float32)
    norm_label[label_no - 1] = 1
    return norm_label

def denormalize_label(
    label: torch.tensor,
    ):
    '''
    de-normalize the output arrays to raw label
    '''
    return torch.argmax(label) + 1

# Method to split training validation test sets
def stratified_split(
    dicoms: Iterable,
    labels: Iterable,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 233,
    ):
    '''
    inputs:
        data: Iterable of the x-data
        labels: Iterable of labels
        validation: 
    '''
    data_dict = {}
    for i, label in enumerate(labels):
        if label not in data_dict:
            data_dict[label] = []
        data_dict[label].append(dicoms[i])
    # size of training set
    training_size = 1 - validation_size - test_size
    if training_size <= 0:
        raise ValueError('training size must be positive')
    # minimum count of data in training, validation and test
    min_training_count = 2
    min_validation_count = 1
    min_test_count = 1
    # shuffle data under each label, and append to the three sets
    training_data, validation_data, test_data, \
        training_labels, validation_labels, test_labels = \
            [], [], [], [], [], []
    for label, label_dicoms in data_dict.items():
        random.Random(seed).shuffle(label_dicoms)
        # i. calculate the sizes of training, validation and testing counts
        # cutoff indices
        total_count = len(label_dicoms)
        validation_count = round(len(label_dicoms) * validation_size)
        test_count = round(len(label_dicoms) * test_size)
        # make sure validation_count and test_count are not zero, when possible
        if total_count - validation_count - test_count > min_training_count + min_validation_count + min_test_count:
            validation_count = max(validation_count, min_validation_count)
            test_count = max(test_count, min_test_count)
        training_count = total_count - validation_count - test_count
        # append dicoms and labels to the lists
        training_data.extend(label_dicoms[:training_count])
        training_labels.extend([label] * training_count)
        validation_data.extend(label_dicoms[training_count: training_count + validation_count])
        validation_labels.extend([label]* validation_count)
        test_data.extend(label_dicoms[training_count + validation_count:])
        test_labels.extend([label] * test_count)
    return training_data, validation_data, test_data, \
        training_labels, validation_labels, test_labels
