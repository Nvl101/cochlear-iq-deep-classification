'''
cochlear implant dataset, with data loading and resizing features

TODO: a mechanism to buffer numpy of image array into tempfiles
TODO: read 
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # to resolve torchvision import problem
import tempfile
from typing import Iterable, Tuple
from re import split
import numpy as np
from pydicom import dcmread
from torchvision import transforms
from torch.utils.data import Dataset
from dataio.utility import normalize_image, normalize_label
from dataio.dataviewer import DataViewer


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
    def __init__(
        self,
        dicom_paths: Iterable[str],
        labels: Iterable[int],
        dicom_dir: str = None,
        img_size: Tuple[int] = (64, 64),
        image_transforms: Iterable[object] = None,
        label_transforms: Iterable[object] = None,
        data_viewer: bool = False,
        ):
        '''
        arguments:
            dicom_dir: str, path of the dicom directory, with sub directories for patient profiles
            dicom_paths: iterable of dicom file paths, joined by the 
            labels: iterable of labels
            img_size: size of the images before inputting to network
            n_classes: number of classes of labels
        '''
        # check that dicom and label count match
        assert len(dicom_paths) == len(labels), \
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
        self.labels = list(labels)
        self.img_size = img_size
        # default image and label transform functions
        default_image_transforms = transforms.Compose([
            transforms.Lambda(normalize_image),
            transforms.ToTensor(),
        ])
        default_label_transforms = transforms.Compose([
            transforms.Lambda(normalize_label),
        ])
        # if image and label transform arguments present,
        # compose onto default transform methods
        if image_transforms:
            self.image_transforms = transforms.Compose([
                default_image_transforms,
                image_transforms, # augmentation before resize, so size doesn't change
                transforms.Resize(self.img_size),
            ])
        else:
            self.image_transforms = transforms.Compose([
                default_image_transforms,
                transforms.Resize(self.img_size),
            ])
        if label_transforms:
            self.label_transforms = transforms.Compose([
                label_transforms,
                default_label_transforms, # normalize label last
            ])
        else:
            self.label_transforms = default_label_transforms
        # buffer dicom image arrays into tempfile
        # make a temp folder to keep all image files, randomly named
        self.buffer_dir = tempfile.mkdtemp()
        buffer_files_list = []        
        for dicom_path in self.dicom_paths:
            image_array = self._read_dicom(dicom_path)
            assert isinstance(image_array, np.ndarray)
            buffer_file_path = tempfile.mktemp(suffix='dcm_', dir = self.buffer_dir)
            np.save(buffer_file_path, image_array)
            buffer_files_list.append(buffer_file_path)
        assert len(buffer_files_list) == len(dicom_paths)
        self.buffer_files = tuple(buffer_files_list)
        # create a data viewer to view the dicom files
        # FIXME: should move to create_dataloader() function
        if data_viewer:
            self.dataviewer = DataViewer(
                self.dicom_paths, self.labels,
                self.dicom_paths.replace('_'.join(os.path.normpath(dicom_paths).split(os.sep))),
                self.buffer_files,
            )
            self.dataviewer.run()
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
        return len(self.labels)
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
            labels: image quailty labels 
        '''
        # image as x-data
        label_raw = self.labels[idx]
        try:
            buffer_path = self.buffer_files[idx]
            image_array = np.load(buffer_path)
        except FileNotFoundError:
            dicom_path = self.dicom_paths[idx]
            image_array = self._read_dicom(dicom_path) # read image array from dicom
        # transforms on image data
        image = self.image_transforms(image_array)
        # label as y-data
        label = self.label_transforms(label_raw)
        return image, label

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
