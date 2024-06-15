'''
manual labelling from the images
'''
import os
import tempfile
import pandas as pd

images_folder = r"D:\data\cochlear-project\image\crop"
output_table_path = r"D:\data\cochlear-project\tables\landmark-3.csv"
labels = ['1', '2', '3']

def main():
    '''
    label images, output tracking table
    '''
    # list images in the images folder, make a table
    tracking_table = pd.DataFrame(
        {'image_filename': os.listdir(images_folder)})
    tracking_table['image_path'] = tracking_table['image_path'].apply(
        lambda x: os.path.join(images_folder, x))
    # detect patient id and dicom no on the images
    tracking_table['patient_id'] = 
    
    # make tempfolder to sort the images
    root_sorting_folder = tempfile.mkdtemp()
    for label in labels:
        os.mkdir(os.path.join(root_sorting_folder, str(label)))
    # 