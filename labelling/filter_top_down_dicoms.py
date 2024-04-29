'''
this script filters out top-down images from the 

TODO: migrate from image_quality_classification.py
'''

import os
import concurrent.futures as cf
import typing
import numpy as np
import pandas as pd
from pydicom import dcmread


data_dir = r"D:\data\cochlear-project\dicom"
tracking_table_path = r"D:\data\cochlear-project\tables\top_down_images_2.csv"
patient_profiles = os.listdir(data_dir)
patient_profiles = ['0' * (5 - len(x)) + x for x in patient_profiles]
patient_profiles.sort()
patient_profiles = [x.lstrip('0') for x in patient_profiles]

def is_top_down_dicom(dicom_file_path: str, max_angle: float = 10.0):
    '''
    determine whether a single dicom file is a 
    '''
    ds = dcmread(dicom_file_path)
    if not hasattr(ds, 'ImageOrientationPatient'):
        return False
    orientation = ds.ImageOrientationPatient
    if orientation is None:
        return False
    if np.dot(orientation[3:], [0, 0, -1]) < np.cos(np.radians(max_angle)):
        return False
    if not hasattr(ds, 'pixel_array'):
        return False
    if ds.pixel_array.ndim > 2:
        return False
    return True

def filter_top_down_dicoms(
        dicom_files: typing.List[str],
        max_angle: float = 10.0,
        mask: typing.List[str] = None,
        ) -> typing.List[str]:
    '''
    filter top-down images
    input:
        `dicom_files`: list of dicom files
        `max_angle`: max angle of image, beyond that will be filtered out
        `mask`: output of the name
    output:
        list of dicom files with top-down views
    '''
    futures_list = []
    # read image orientation using
    with cf.ThreadPoolExecutor(max_workers=16, thread_name_prefix='dicom_filter_') as executor:
        for dicom_file in dicom_files:
            futures_list.append(executor.submit(is_top_down_dicom, dicom_file, max_angle))
    result_list = [future.result() for future in futures_list]
    if mask is None:
        mask = dicom_files
    top_down_files = []
    for dicom_file, is_top_down in zip(mask, result_list):
        if is_top_down:
            top_down_files.append(dicom_file)
    return top_down_files


def main():
    '''
    main process: filter top down images, write to a table
    '''
    patient_ids = []
    dicom_filenames = []
    dicom_paths = []

    n_total, n_filtered = 0, 0

    print('filtering top-down images')

    for patient_profile in patient_profiles:
        print(f'- patient {patient_profile}...')
        patient_dir = os.path.join(data_dir, patient_profile, 'DICOM')
        all_dicom_filenames = os.listdir(patient_dir)
        all_dicom_paths = [os.path.join(patient_dir, filename) for filename in all_dicom_filenames]
        filtered_dicom_filenames = filter_top_down_dicoms(all_dicom_paths, 10, all_dicom_filenames)
        filtered_dicom_paths = [os.path.join(patient_profile, 'DICOM', dicom_filename) \
                               for dicom_filename in filtered_dicom_filenames]
        # extending id, filename and file paths
        patient_id = [patient_profile] * len(filtered_dicom_filenames)
        patient_ids.extend(patient_id)
        dicom_filenames.extend(filtered_dicom_filenames)
        dicom_paths.extend(filtered_dicom_paths)
        # updating total
        n_patient_total = len(all_dicom_filenames)
        n_patient_filtered = len(filtered_dicom_filenames)
        n_total += n_patient_total
        n_filtered += n_patient_filtered
        print(f'  {n_patient_filtered} out of {n_patient_total}')

    tracking_table = pd.DataFrame(
        {
            'patient_id': patient_ids,
            'dicom_filename': dicom_filenames,
            'dicom_path': dicom_paths,
        }
    )
    tracking_table.to_csv(tracking_table_path)
    print(f'total: {n_filtered} out of {n_total}')
    print(f'written to csv: {tracking_table_path}')

if __name__ == '__main__':
    main()
