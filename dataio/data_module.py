'''
TODO: sync updates from dataloader.py
including sampler functions

data module for classifier
1. splits training, validation and test data
2. create training, validation and test datasets and dataloaders
3. method to create prediction dataset
'''
from abc import abstractmethod
from typing import Iterable, Tuple
from torch.utils.data import Dataset, DataLoader
from .dataset import CochlearIqDataset
from .utility import stratified_split, grouped_split


class DataModule:
    training_dataset: object
    validation_dataset: object
    test_dataset: object
    training_dataloader: object
    validation_dataloader: object
    test_dataloader: object

    @abstractmethod
    def prediction_dataloader(self) -> Tuple[Dataset, DataLoader]:
        '''
        method to obtain prediction dataset and dataloader
        '''
        raise NotImplementedError()


class CochlearIqDataModule(DataModule):
    '''
    Data module for cochlear image quality datasets

    NOTE: simply migrating methods from `dataloader.py` to here, added `prediction_dataloader` method
    '''
    target_img_size: tuple = (64, 64)
    def __init__(
            self,
            dicom_paths: Iterable[str],
            labels: Iterable[int],
            dicom_dir: str = None,
            groups: Iterable[str] = None,
            **kwargs,
        ):
        '''
        initiate by giving the benchmark dataset,
        the object will split the 

        arguments:
            dicom_paths: list of dicom file path strings
        '''
        # resolving argument for configurations
        validation_size = kwargs.get('validation_size', 0.1)
        test_size = kwargs.get('test_size', 0.1)
        seed = kwargs.get('seed', 233)
        batch_size = kwargs.get('batch_size', 1)
        num_workers = kwargs.get('num_workers', 16)
        # split data into training, validation and testing sets
        if groups is None:
            # if group is given (by patient_id), the patient profiles are split by groups
            training_dicoms, validation_dicoms, test_dicoms, \
            training_labels, validation_labels, test_labels = \
                stratified_split(dicom_paths, labels, validation_size, test_size, seed)
        else:
            # if groups not given, stratify split dataset randomly
            training_dicoms, validation_dicoms, test_dicoms, \
            training_labels, validation_labels, test_labels = \
                grouped_split(dicom_paths, labels, groups, validation_size, test_size, seed)
        # creating datasets and dataloaders
        self.training_dataset = CochlearIqDataset(training_dicoms, training_labels, dicom_dir)
        self.validation_dataset = CochlearIqDataset(validation_dicoms, validation_labels, dicom_dir)
        self.test_dataset = CochlearIqDataset(test_dicoms, test_labels, dicom_dir)
        # defining the dataloaders
        self.training_dataloader = DataLoader(self.training_dataset, batch_size, num_workers=num_workers, shuffle=True)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size, num_workers=num_workers)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size, num_workers=num_workers)

    def prediction_dataloader(self, dicoms, dicom_dir = None):
        '''
        method to create dataset and dataloader for 
        '''
        # create a dataset with only dicoms no labels
        pred_dataset = CochlearIqDataset(dicoms, None, dicom_dir)
        pred_dataloader = DataLoader(pred_dataset, 1, 16)
        return pred_dataset, pred_dataloader
