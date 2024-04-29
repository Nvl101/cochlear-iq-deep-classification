'''
dataloader: takes dataset through transformation

DEPRECATED: now using data_module.py to create dataloaders
this allows creating prediction dataset dynamically
'''
# import os # using relative import from file path
# script_dir = os.path.join(__file__, '..', '..')
import os
from typing import Iterable
from torch.utils.data import DataLoader
from .dataset import CochlearIqDataset
from .utility import stratified_split
from .data_sampler import RandomOverSampler
from .dataviewer import DataViewer

DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = 16

def from_training_data(
    dicoms: Iterable[str],
    labels: Iterable,
    batch_size = 1,
    num_workers = 16
    ):
    '''
    dataset and dataloader from training data

    inputs:
        `dicoms`: paths of the dicom files
    '''
    training_dataset = CochlearIqDataset(dicoms, labels, dicom_dir)
    training_dataloader = DataLoader(training_dataset, batch_size, num_workers=num_workers, shuffle=True)
    return {
        'dataset': training_dataset,
        'dataloader': training_dataloader
    }

def from_image_only(
    dicoms: Iterable[str],
    ):
    dataset = CochlearIqDataset(dicoms, dicom_dir)


def create_dataloader(
    dicoms: Iterable[str],
    labels: Iterable,
    dicom_dir: str = None,
    batch_size: int = 4,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    num_workers: int = 16,
    img_size: tuple = (64, 64),
    use_3_channels: bool = False,
    max_duplicates: int = 10,
    *args, **kwargs,
    ):
    '''
    inputs:
        `dicoms`: paths of dicom files
        `labels`: image quality labels
        `dicom_dir`: directory containing the patient DICOM files
        `batch_size`: data to load from the Dataset each time
        `validation_ratio`: float, ratio of validation data in the whole dataset
        `test_size`: float, ratio of test data in the whole dataset
        `view_validation`: whether to create DataViewer for validation set
        `view_test`: whether to create DataViewer for test set
    outputs:
        dict, may contain keys as follows:
        
        'training_dataset', 'validation_dataset', 'test_dataset',
        'training_dataloader', 'validation_dataloader', 'test_dataloader',
        'validation_dataviewer', 'test_dataviewer'
    '''
    # splitting the x, y data with stratified method
    training_dicoms, validation_dicoms, test_dicoms, \
        training_labels, validation_labels, test_labels = \
            stratified_split(dicoms, labels, validation_size, test_size)
    # initiating the datasets
    training_dataset = CochlearIqDataset(
        training_dicoms, training_labels, dicom_dir,
        img_size=img_size, augmentation=True, use_3_channels=use_3_channels)
    validation_dataset = CochlearIqDataset(
        validation_dicoms, validation_labels, dicom_dir,
        img_size=img_size, use_3_channels=use_3_channels)
    test_dataset = CochlearIqDataset(
        test_dicoms, test_labels, dicom_dir,
        img_size=img_size, use_3_channels=use_3_channels)
    # sampler for training dataset
    training_sampler = RandomOverSampler(training_labels, max_duplicates=max_duplicates)
    # defining the dataloaders
    training_dataloader = DataLoader(training_dataset, batch_size, num_workers=num_workers, sampler=training_sampler)
    validation_dataloader = DataLoader(validation_dataset, 1, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, 1, num_workers=num_workers)
    # dataviewer for augmented training set
    # training_dataviewer = DataViewer.from_dataset(training_dataset)
    return_dict = {
        'training_dataset': training_dataset,
        'validation_dataset': validation_dataset,
        'test_dataset': test_dataset,
        'training_dataloader': training_dataloader,
        'validation_dataloader': validation_dataloader,
        'test_dataloader': test_dataloader,
        'sampler': training_sampler,
        # 'training_dataviewer': training_dataviewer,
        'dataloaders': (training_dataloader, validation_dataloader, test_dataloader),
        'datasets': (training_dataset, validation_dataset, test_dataset),
    }
    return return_dict

# a function to simply make prediction dataloader
def create_pred_dataloader(dicoms: Iterable[str], dicom_dir: str = None, img_size: tuple=(64,64), use_3_channels: bool = False):
    '''
    a function to simply make prediction dataloader
    '''
    if dicom_dir is not None:
        prediction_dicoms = [os.path.join(dicom_dir, dicom) for dicom in dicoms]
    else:
        prediction_dicoms = dicoms
    prediction_dataset = CochlearIqDataset(prediction_dicoms, img_size=img_size, use_3_channels=use_3_channels)
    prediction_dataloader = DataLoader(prediction_dataset, batch_size=1, num_workers=DEFAULT_NUM_WORKERS)
    return prediction_dataloader

if __name__ == '__main__':
    # TODO: following code will be integrated
    # HARDCODE: arguments
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
    from config import tracking_table, dicom_dir
    # definition of dataloader
    dicoms = tracking_table['dicom_path'].to_list()
    labels = tracking_table['label'].astype('int16').to_list()
    training_dicoms, validation_dicoms, test_dicoms, \
        training_labels, validation_labels, test_labels = \
            stratified_split(dicoms, labels)
    training_dataset = CochlearIqDataset(training_dicoms, training_labels, dicom_dir)
    validation_dataset = CochlearIqDataset(validation_dicoms, validation_labels, dicom_dir)
    test_dataset = CochlearIqDataset(test_dicoms, test_labels, dicom_dir)
    sample_data = training_dataset[0]
    training_dataloader = DataLoader(training_dataset, 4, num_workers=8, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, 4, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, 4)
    print('debug...')
