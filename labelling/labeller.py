'''
labeller class

which handles the manual or semi-supervised labelling process
'''
import os
from abc import abstractclassmethod
import concurrent.futures as cf
import tempfile
from typing import Iterable
import pandas as pd
from dataio.utility import read_dicom_image, normalize_image, write_image


# utility functions
def dicoms_to_imgs(dicom_paths: Iterable[str], img_paths: Iterable[str]):
    '''
    read dicom files, and write to images in multithread
    inputs:
        dicom_paths: paths of the dicom files
        img_paths: target path of the images
    operations:
        write images to corresponding target paths
    '''
    def dicom_to_img(dicom_path: str, img_path: str):
        # function that read from a dicom, writes image into target path
        # supports multi-threading
        img_array = read_dicom_image(dicom_path)
        norm_img_array = normalize_image(img_array)
        write_image(norm_img_array, img_path)
    assert len(dicom_paths) == len(img_paths)
    futures_list = []
    with cf.ThreadPoolExecutor(max_workers=16) as executor:    
        for dicom_path, img_path in zip(dicom_paths, img_paths):
            futures_list.append(executor.submit(dicom_to_img, dicom_path, img_path))
    for future in futures_list:
        if future.exception():
            raise future.exception()

def update_img_labels(tempdir: str, label_folders: Iterable[str]=None):
    '''
    update the sorting directory
    input:
    - tempdir: temp directory path
    labels
    output:
    - image_labels: a list of (image, label)
    '''
    if label_folders is None:
        label_folders = []
        for folder_name in os.listdir(tempdir):
            if os.path.isdir(os.path.join(tempdir, folder_name)):
                label_folders.append(folder_name)
        label_folders = [os.path.join(tempdir, folder_name) for folder_name in label_folders]
    label_dir_paths = [os.path.join(tempdir, str(label)) for label in label_folders]
    image_labels = []
    for label, folder in zip(label_folders, label_dir_paths):
        images_under_label = [(file, label) for file in os.listdir(folder) \
                if os.path.isfile(os.path.join(folder, file)) \
                and file.endswith('.png')]
        image_labels.extend(images_under_label)
    return image_labels


class abstractLabeller:
    '''
    labeller class that handles semi-supervised labelling process
    contains corresponding:
        - semi-supervised model
        - label tracking and sorting object
    '''
    labeller_model: object # contain train, evaluate, predict methods
    sorter: object # for tracking the 
    def __init__(self, *args, **kwargs):
        pass
    @abstractclassmethod
    def release_images(self, *args, **kwargs):
        pass
    @abstractclassmethod
    def update_labels(self, *args, **kwargs):
        pass


class abstractSorter:
    @abstractclassmethod
    def clear_sorting_folder(self, *args, **kwargs):
        pass
    @abstractclassmethod
    def release_images(self, *args, **kwargs):
        pass
    @abstractclassmethod
    def track_labels(self, *args, **kwargs):
        pass


class Sorter(abstractSorter):
    '''
    class that responsible for 
    '''
    sorting_folder: str
    labels: Iterable[str] # all possible labels
    tracking_table: pd.DataFrame # tracks dicom paths and actual labels
    
    def __init__(self, tracking_table, labels=None):
        self.tracking_table = tracking_table
        self.labels = tracking_table['labels'].unique() if labels is None else labels
        self.sorting_folder = tempfile.mkdtemp('ciiq_sorter_')

    def clear_sorting_folder(self, ):
        '''
        clean up images in the sorting folder
        '''
        label_dir_paths = [os.path.join(self.sorting_folder, label) for label in self.labels]
        for folder in [self.sorting_folder, *label_dir_paths]:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path): # delete files not folders
                    os.remove(file_path)

    def release_images(self, dicom_paths: Iterable[str], labels: Iterable[str] = None):
        dicom_basename = lambda dicom_path: os.path.basename(dicom_path)
        if labels is None:  # if labels not given, release all images to the root sorting folder
            target_paths = []
            for dicom_path in dicom_paths:
                target_path = os.path.join(self.sorting_folder, dicom_basename(dicom_path))
                target_paths.append(target_path)
            target_paths = self.sorting_folder
        else:  # if labels given, release to corresponding label folders
            # check the paths and labels
            assert len(dicom_paths) == labels, 'length of dicom paths and labels not match'
            # iterate through the labels, to get target paths
            for dicom_path in dicom_paths:
                target_path = os.path.join(self.sorting_folder, )
        pass

    def track_labels(self):
        '''
        track labels inside the sorting folder
        returns:
        '''
        if label_folders is None:
            label_folders = []
            for folder_name in os.listdir(tempdir):
                if os.path.isdir(os.path.join(tempdir, folder_name)):
                    label_folders.append(folder_name)
        label_folders = [os.path.join(self.sorting_folder, folder_name) for folder_name in label_folders]
        label_dir_paths = [os.path.join(self.sorting_folder, str(label)) for label in label_folders]
        image_labels = []
        for label, folder in zip(label_folders, label_dir_paths):
            images_under_label = [(file, label) for file in os.listdir(folder) \
                    if os.path.isfile(os.path.join(folder, file)) \
                    and file.endswith('.png')]
            image_labels.extend(images_under_label)
        return image_labels

class Labeller(abstractLabeller):
    '''
    object that handles the labelling of image quality

    labelling process:
    1. read a tracking table, which contains three fields:
        - patient_id: str, patient id which the dicom belongs to
        - dicom_path: path of the dicom file
        - label: str, label if the dicom image, null if not labelled yet
    2. release dicom images into the sorting folder
        - if labelling model available, predict with the model
    3. wait user to manually correct labels
    4. update the tracking table with labels

    '''
    def __init__(
            self, source_tracking_table: str,
            target_tracking_table: str,
            labelling_model: object = None
            ):
        '''
        arguments:
        - source_tracking_table: path to input the tracking table
        - target_tracking_table: path to output the updated tracking table
        - labelling_model: the model used to predict and train on labels
        '''
        # read labels from tracking table
        self.tracking_table = pd.read_csv(source_tracking_table)
        pass

    def _read_tracking_table(self, tracking_table_path: str, dicom_path: str):
        assert os.path.isdir(dicom_path)
        tracking_table = pd.read_csv(tracking_table_path)
        self.dicom_paths = tracking_table['dicom_paths']
        self.labels = tracking_table['labels']

    def _dicom_to_imgs(dicom_path: Iterable[str], img_labels: Iterable[str] = None):
        '''
        write dicom images to folders of their corresponding labels
        '''
        pass

    def _predict_labels(dicom_paths: Iterable[str]):
        '''
        predict the labels of 

        for manual labelling, simply output None
        '''
        labels = self.labelling_model

    def release_images(self, *args, **kwargs):
        # TODO: write method
        pass
    
    def update_labels(self, *args, **kwargs):
        # TODO: write method
        pass


class ManualLabeller:
    def __init__(self, *args, **kwargs):
        self.super(*args, **kwargs).__init__()
    def _predict_labels(*args, **kwargs):
        return None
    

