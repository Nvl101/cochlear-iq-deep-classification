'''
cochlear implant dataset, with data loading and resizing features

TODO: a mechanism to buffer numpy of image array into tempfiles
TODO: read 
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # to resolve torchvision import problem
import tempfile
from typing import Any, Iterable, List, Tuple
from re import split
import numpy as np
from pydicom import dcmread
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from dataio.transforms import grayscale_to_rgb, truncate_thresholding
from dataio.utility import normalize_image, normalize_label, clip_image


def _grayscale_to_rgb(image: torch.Tensor):
    '''
    transforms 1-channel grayscale image to rgb
    returns:
        - image with 3 channels
    '''
    if image.shape[0] == 1:
        return image.repeat([3, 1, 1])
    else:
        return image

# the dataset class
class CochlearIqDataset(Dataset):
    '''
    cochlear image quality dataset, containing images and labels
    '''
    dicom_paths: Iterable[str]
    buffer_paths: Iterable[str] # path of buffered image pixel array
    labels: Iterable[int]
    img_size: Tuple[int]
    dicom_dir: str = None
    # temporary files to buffer dicom image arrays
    buffer_dir: str = None
    buffer_files: Tuple[str]
    data_viewer: object = None
    # statistics of label counts
    _label_count: dict = None
    def __init__(
        self,
        dicom_paths: Iterable[str],
        labels: Iterable[int] = None,
        dicom_dir: str = None,
        img_size: Tuple[int] = (64, 64),
        augmentation: bool = False,
        use_3_channels: bool = False,
        dtype: torch.dtype = torch.float32,
        *args, **kwargs
        ):
        '''
        arguments:
            dicom_dir: str, path of the dicom directory, with sub directories for patient profiles
            dicom_paths: iterable of dicom file paths, joined by dicom_dir to get ful path 
            labels: iterable of labels, by default None, which dataset only outputs images
            img_size: size of the images before inputting to network
            n_classes: number of classes of labels
            use_3_channels: setting True will transform grayscale image to 3-channel RGB
            dtype: output image arrays and labels will convert to this datatype
        '''
        # check that dicom and label count match
        assert labels is None or len(dicom_paths) == len(labels), \
            'lengths of dicom paths and labels not equal: {len(dicom_paths)}, {len(labels)}'
        norm_dicom_paths = (os.path.join(*split(r'[\\/]', dicom_path)) \
            for dicom_path in dicom_paths)
        # join dicom_dir if present, else deem as complete dicom file path
        if dicom_dir:
            # check that dicom_dir exists, then join the dicom files
            if not os.path.isdir(dicom_dir):
                raise FileNotFoundError(f'dicom directory {dicom_dir} not found')
            self.dicom_paths = [os.path.join(dicom_dir, dicom_path) \
                for dicom_path in norm_dicom_paths]
        else:
            self.dicom_paths = [dicom_path for dicom_path in norm_dicom_paths]
        self.labels = list(labels) if labels is not None else None
        self.img_size = img_size
        self._use_3_channels = use_3_channels
        self.dtype = dtype
        # get image and label transformer
        self.image_transforms = self._get_image_transformer(augmentation)
        if self.labels is not None:
            self.label_transforms = self._get_label_transformer() 
        else:
            self.label_transforms = None
        # statistics for label counts
        self._label_count = dict()
        if self.labels is not None:
            for label in labels:
                if label in self._label_count:
                    self._label_count[label] += 1
                else:
                    self._label_count[label] = 0
        # buffer dicom image arrays into tempfile
        # make a temp folder to keep all image files, randomly named
        self.buffer_dir = tempfile.mkdtemp(prefix='ciiq_dataset_')
        buffer_files_list = []        
        for dicom_path in self.dicom_paths:
            image_array = self._read_dicom(dicom_path)
            assert isinstance(image_array, np.ndarray)
            buffer_file_path = tempfile.mktemp(prefix='buffer_', suffix='.npy', dir = self.buffer_dir)
            np.save(buffer_file_path, image_array)
            buffer_files_list.append(buffer_file_path)
        assert len(buffer_files_list) == len(dicom_paths)
        self.buffer_files = tuple(buffer_files_list)

    def _get_image_transformer(self, image_augmentation: bool = False):
        # default transform methods, applied on all images
        # default_image_transforms = transforms.Compose([
        #     transforms.Lambda(normalize_image),
        #     transforms.ToTensor(),
        # ])

        lst_image_transforms = [
            # starts with numpy array image
            transforms.Lambda(normalize_image),
            # transforms.Lambda(truncate_thresholding) if image_augmentation else None,
            transforms.Lambda(clip_image),
            transforms.ToTensor(),
            transforms.Lambda(grayscale_to_rgb) if self._use_3_channels else None,
            # transforms.Resize((256, 256)),
            # transforms.CenterCrop((224, 224)) if image_augmentation else None,
            transforms.RandomHorizontalFlip(p=0.5) if image_augmentation else None,
            transforms.RandomResizedCrop(
                size=self.img_size, scale=(1.0, 1.1), ratio=(0.9, 1.0)) if image_augmentation else None,
            transforms.Resize(self.img_size),
        ]

        lst_image_transforms = [x for x in lst_image_transforms if x is not None]

        image_transformer = transforms.Compose(lst_image_transforms)

        ### CUTOVER: new method above switches list elements by if statement

        # lst_default_transforms = [
        #     transforms.Lambda(normalize_image),
        #     transforms.ToTensor(),
        # ]
        # if self._use_3_channels:
        #     lst_default_transforms.append(_grayscale_to_rgb)
        # default_image_transforms = transforms.Compose(lst_default_transforms)
        # # augmentation transform methods
        # # random_flip = transforms.RandomHorizontalFlip(p=0.5)
        # # padded_crop = transforms.RandomCrop(size=68, padding=4, probability=(0.8, 0.8, 0.1, 0.15))
        # # center_crop = transforms.CenterCrop(size=self.img_size)
        # # random_rotation = transforms.RandomRotation(degrees=7)
        # # random_resized_crop = transforms.RandomResizedCrop(
        # #         size=self.img_size, scale=(0.9, 1.0), ratio=(0.8, 0.9))
        # image_augmentation_transforms = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     # transforms.RandomRotation(degrees=7),
        #     transforms.RandomResizedCrop(
        #         size=self.img_size, scale=(1.0, 1.1), ratio=(0.9, 1.0)),
        # ])
        # if image_augmentation:
        #     image_transformer = transforms.Compose([
        #         default_image_transforms,
        #         transforms.Lambda(random_truncate_thresholding),
        #         image_augmentation_transforms, # augmentation before resize, so size doesn't change
        #         # transforms.Resize(self.img_size),
        #     ])
        # else:
        #     image_transformer = transforms.Compose([
        #         default_image_transforms,
        #         transforms.Resize(self.img_size),
        #     ])
        return image_transformer
    
    def _get_label_transformer(self):
        default_label_transforms = transforms.Compose([
            transforms.Lambda(normalize_label),
        ])
        label_transformer = default_label_transforms
        # if label_transforms:
        #     label_transformer = transforms.Compose([
        #         label_transforms,
        #         default_label_transforms, # normalize label last
        #     ])
        # else:
        return label_transformer
    
    # dataset loading functions
    def _read_dicom(self, dicom_path):
        '''
        read dicom file and return image array
        '''
        if not os.path.isfile(dicom_path):
            raise FileNotFoundError(f'dicom file {dicom_path} not found')
        ds = dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype('int16')
        return pixel_array
    def __len__(self):
        if self.labels is not None:
            assert len(self.labels) == len(self.dicom_paths)
        return len(self.dicom_paths)
    def __getraw__(self, idx):
        '''
        debug function, to get raw image and labels
        '''
        dicom_path = self.dicom_paths[idx]
        label = self.labels[idx]
        return dicom_path, label
    def __getitem__(self, idx):
        '''
        input:
            idx: int, index of the tracking table
        output:
            image: image pixel array
            (if the dataset has labels)
            labels: image quailty labels 
        '''
        # image as x-data
        try:
            buffer_path = self.buffer_files[idx]
            image_array = np.load(buffer_path)
        except FileNotFoundError:
            dicom_path = self.dicom_paths[idx]
            image_array = self._read_dicom(dicom_path) # read image array from dicom
        # transforms on image data
        image = self.image_transforms(image_array)
        image = image.to(self.dtype)
        # label as y-data
        if self.labels is not None:
            label_raw = self.labels[idx]
            label = self.label_transforms(label_raw)
            label = label.to(self.dtype)
            return image, label
        else:
            return image

class LandmarkImages(Dataset):
    '''
    dataset of landmarked cochlear implantation images,
    which are detected by YOLO and focus on the cochlear spiral
    '''
    image_paths: Iterable[str]
    output_size: Tuple[int] = (256, 256)
    _use_3_channels: bool = True
    def __init__(self, image_paths: Iterable, output_size=(256, 256)):
        self.image_paths = image_paths
        self.output_size = output_size
        self.image_transformer = self._get_image_transformer(augmentation=True)
    def __len__(self):
        return len(self.image_paths)
    def _load_image(self, image_path):
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f'image file not found: {image_path}')
        image_array = cv2.imread(image_path)
        return image_array
    def __getitem__(self, idx):
        image_array = self._load_image(self.image_paths[idx])
        image_array_aug = self.image_transformer(image_array)
        return image_array_aug
    def _get_image_transformer(self, augmentation: bool = False):
        lst_image_transforms = [
            transforms.Lambda(normalize_image),
            transforms.ToTensor(),
            transforms.Lambda(grayscale_to_rgb) if self._use_3_channels else None,
            transforms.RandomHorizontalFlip(p=0.5) if augmentation else None,
            transforms.RandomResizedCrop(
                size=self.output_size, scale=(1.0, 1.1), ratio=(0.8, 1.0))
                if augmentation else None,
            transforms.Resize(self.output_size),
        ]
        lst_image_transforms = [x for x in lst_image_transforms if x is not None]
        image_transformer = transforms.Compose(lst_image_transforms)
        return image_transformer
    
class ImageQualityLabels(Dataset):
    labels: Iterable[int]
    possible_labels: Iterable[int] = [1, 2, 3]
    def __init__(self, labels: Iterable):
        self.labels = [int(label) for label in labels]
        for label in self.labels:
            assert label in self.possible_labels, 'label not in possible labels'
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        raw_label = self.labels[idx]
        return self._normalize_label(raw_label)
    def _normalize_label(self, label: int):
        lst_label = [.0, .0, .0]
        lst_label[label - 1] = 1.0
        return torch.Tensor(lst_label)

class ImageLabels(Dataset):
    '''
    dataset of images and corresponding labels
    '''
    def __init__(self, images: LandmarkImages, labels: ImageQualityLabels):
        assert len(images) == len(labels), 'length of images and labels not equal'
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# debug
if __name__ == '__main__':
    # import sys
    # sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
    # import pandas as pd
    # from config import tracking_table_path
    # # initiate a cochlear iq dataset
    # data_dir = os.path.normpath(os.path.abspath(os.path.join(__file__, '..','..','..')))
    # dicom_dir = os.path.join(data_dir, 'dicom')
    # assert os.path.isfile(tracking_table_path), \
    #     'tracking table {tracking_table_path} does not exist'
    # df_detectability = pd.read_csv(tracking_table_path)
    # dicom_paths, labels = df_detectability['dicom_path'], df_detectability['label']
    # dataset = CochlearIqDataset(data_dir, dicom_paths, labels)
    # sample_data = dataset[5]
    print('debug...')
