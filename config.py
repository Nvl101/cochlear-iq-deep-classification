'''
configuration for deep image quailty classification

under USER SETTINGS are user configurable 
'''
import os
from pandas import read_csv

## USER SETTINGS
# dataset path configurations
project_dir = os.path.abspath(
    os.path.join(__file__, os.path.normpath('../../..')))
# data_dir = os.path.join(project_dir, 'data') # on IM2
data_dir = "D:\\data\\cochlear-project"# on Foundry Computer
dicom_dir = os.path.join(data_dir, 'dicom') # path of dicom files
tracking_table_path = os.path.join(data_dir, 'tables', 'image_detectability_3.csv') # path of tracking table

# validation and test sizes
validation_size = 0.2
test_size = 0.1
# training runtime
n_epoches = 5
learning_rate = 0.001

## OPERATIONS
# checking if the paths exist
assert os.path.isdir(project_dir), f'project_dir {project_dir} does not exist'
assert os.path.isdir(data_dir), f'data_dir {data_dir} does not exist'
assert os.path.isfile(tracking_table_path)
tracking_table = read_csv(tracking_table_path)

# dicom paths and labels
# joining DICOM paths to obtain full paths
# dicom_paths = tracking_table['dicom_path'].to_list()
patient_ids = tracking_table['patient_id'].astype('str').to_list()
dicom_paths = tracking_table.apply(
    lambda x: os.path.join(dicom_dir, str(x.patient_id), 'DICOM', f'I{x.dicom_no}'), axis=1
    ).to_list()
labels = tracking_table['label'].astype('int16').to_list()

if __name__ == '__main__':
    print('debug...')
