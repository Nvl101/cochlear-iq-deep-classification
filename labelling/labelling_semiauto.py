'''
expand image quality labels using trained model

modified from label_img_quality.py
1. model makes prediction for image labels, release into the folder
2. user will review and make adjustments
3. record image quaity in a 
'''
# imports
import os
project_dir = os.path.abspath(os.path.join(__file__, '..','..', 'deep-classification','..'))
import sys
sys.path.insert(1, project_dir)
import concurrent.futures as cf
import subprocess
import tempfile
from typing import Iterable
import pandas as pd
import torch
from dataio.dataloader import create_pred_dataloader
from dataio.utility import read_dicom_image, normalize_image, write_image
# from networks.resnet_classifier import resnet18_classifier
from networks.densenet_classifier import densenet121_classifier


# configurations
labels = ['1', '2', '3']

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

def clean_sorting_folders(root_folder: str, label_dirs: Iterable[str]=None):
    '''
    clean up sorting temp directory
    '''
    if label_dirs is None:
        label_dirs = os.listdir(root_folder)
    label_dir_paths = [os.path.join(root_folder, label_dir) for label_dir in label_dirs]
    bin_paths = [os.path.join(root_folder, label_dir, 'bin') for label_dir in label_dirs]
    for folder in [root_folder, *label_dir_paths, *bin_paths]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if file_path.endswith('.png'): # delete png files
                os.remove(file_path)

def update_img_labels(tempdir: str, label_folders: Iterable[str]=None):
    '''
    update the sorting directory
    input:
    - tempdir: temp directory path
    - 
    '''
    if label_folders is None:
        label_folders = []
        for folder_name in os.listdir(tempdir):
            if os.path.isdir(os.path.join(tempdir, folder_name)):
                label_folders.append(folder_name)
        label_folders = [os.path.join(tempdir, folder_name) for folder_name in label_folders]
    label_dir_paths = [os.path.join(tempdir, str(label)) for label in label_folders]
    label_dir_paths.extend([os.path.join(tempdir, str(label), 'bin') for label in label_folders])
    image_labels = []
    for label, folder in zip(label_folders, label_dir_paths):
        images_under_label = [(file, label) for file in os.listdir(folder) \
                if os.path.isfile(os.path.join(folder, file)) \
                and file.endswith('.png')]
        image_labels.extend(images_under_label)
    return image_labels

def main():
    # 1. get top-down dicoms from table
    # mapping table filters out top-down images
    mapping_table_input = r"D:\data\cochlear-project\tables\top_down_images_2-light.csv"
    mapping_table_output = r"D:\data\cochlear-project\tables\image_detectability_2-demo.csv"

    dicom_dir = r"D:\data\cochlear-project\dicom"
    mapping_table = pd.read_csv(mapping_table_input) # maps patient and dicom to path
    # DEBUG: shortening mapping table patient profiles
    # mapping_table = mapping_table[mapping_table['patient_id'].isin(mapping_table['patient_id'].unique()[:3])]

    # 2. create a temp folder with subdirectories named by labels
    tempdir = tempfile.mkdtemp()
    print(tempdir)
    labeldir = [os.path.join(tempdir, str(label)) for label in labels]
    for dir in labeldir: # with corresponding label subdirectories
        os.mkdir(dir)
        os.mkdir(os.path.join(dir, 'bin'))

    # TODO: load model from the binary file
    model_path = r"bin\dense121_state_04-23-ep_15.pkl"
    assert os.path.isfile(model_path)
    state_dict = torch.load(model_path)
    # model = resnet18_classifier
    model = densenet121_classifier
    model.load_state_dict(state_dict)

    # 3. rename according to patient id and original file names, map into temp folder
    #     - keep a mapping table
    #     - create label column
    # comes to the labelling stage:
    # mapping dicom path to absolute path
    extend_full_path = lambda x: os.path.join(dicom_dir, x)
    mapping_table['full_dicom_path'] = mapping_table['dicom_path'].apply(extend_full_path)
    # create tempdir, and map dicom files to target images in tempdir
    # map images to corresponding paths in temp folder
    name_png_file = lambda x: f'P{x.patient_id}-{x.dicom_filename}.png'
    mapping_table['img_name'] = mapping_table.apply(name_png_file, axis=1)
    
    # 4. for each patient,
    #     - place all images under temp folder
    #     - wait for user to finish sorting
    #     - list label folders to update tracking table
    #     - update into the tracking table
    tracking_tables = [] # save the merged table containing label info
    for patient_id, patient_mapping_table in mapping_table.groupby('patient_id'):
        # clean up folders
        clean_sorting_folders(tempdir, labeldir)
        # release images into label directories
        # full_img_path = patient_mapping_table['img_name'].apply(lambda x: os.path.join(tempdir, x))
        pred_labels = []
        full_img_path = []
        # use model to predict image quality label
        print('  fitting...', end='\r')
        prediction_dicoms = patient_mapping_table['dicom_path']
        prediction_dataloader = create_pred_dataloader(
            prediction_dicoms, dicom_dir, img_size=(256, 256), use_3_channels=True)
        for image_array in prediction_dataloader:
            norm_label = model(image_array)
            label = torch.argmax(norm_label).item() + 1
            pred_labels.append(label)
        # releasing into corresponding temp folders for user review
        print('  releasing...', end='\r')
        for img_name, pred_label in zip(patient_mapping_table['img_name'], pred_labels):
            full_img_path.append(os.path.join(tempdir, str(pred_label), img_name))
        dicoms_to_imgs(patient_mapping_table['full_dicom_path'], full_img_path)
        # press Enter to continue
        print("released images for patient", patient_id)
        subprocess.Popen(f'explorer /select,"{labeldir[0]}"')
        input("press Enter to continue...")
        # track image labelling
        # left merge onto original tracking table
        image_labels = update_img_labels(tempdir, labels)
        left_table = patient_mapping_table.copy()
        right_table = pd.DataFrame(image_labels, columns=['img_name', 'label'])
        merge_table = pd.merge(left_table, right_table, how= 'left', \
                            on='img_name', validate='one_to_one')
        # merge_table = update_img_labels(tempdir)
        tracking_tables.append(merge_table)
    # 5. after finishing all patients, release all images into corresponding folders
    tracking_table_labelled = pd.concat(tracking_tables)
    # tracking_table_labelled = merge_table.copy()
    tracking_table_labelled = tracking_table_labelled.reset_index(drop=True)
    # clean up folders
    clean_sorting_folders(tempdir, labeldir)
    # release all images into labelling folders
    labelled_img_paths = tracking_table_labelled.apply(
        lambda x: os.path.join(tempdir, str(x.label), x.img_name), axis=1)
    dicoms_to_imgs(tracking_table_labelled['full_dicom_path'], labelled_img_paths)
    subprocess.Popen(f'explorer /select,"{labeldir[0]}"')
    input("check folder for labelled images")
    # save csv to target path
    tracking_table_labelled = tracking_table_labelled[['patient_id', 'dicom_path', 'label']]
    tracking_table_labelled.to_csv(mapping_table_output, index=False)
    print(f'written to csv: {mapping_table_output}')

if __name__ == '__main__':
    main()
