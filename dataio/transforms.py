'''
augmentation transforms
for cochlear image quality dataset

data augmentation strategies:
random crop, rotation, brightness change, 
'''
import numpy as np
import torch
from torchvision import transforms


# image transform functions
def normalize_image(
    image_array: np.ndarray,
    datatype: str = 'float32',
    upper_bound: float = 3071, # hounsfield upper bound in theory
    lower_bound: float = 0
    ):
    '''
    normalize image array using maximum and minimum values
    but still capped with lower and upper bounds according to hounsfield

    input:
        image_array, integers ranging from 0 to over 4k

    output:
        norm_image, ranging from 0.0 to 1.0
    '''
    # limiting the image upper and lowe bound
    image_array_copy = image_array.astype(datatype)
    upper_bound = max(np.max(image_array_copy), upper_bound)
    lower_bound = min(np.min(image_array_copy), lower_bound)
    img_delta = upper_bound - lower_bound
    norm_image = image_array_copy - lower_bound
    norm_image = norm_image / img_delta
    norm_image = norm_image.astype(datatype)
    return norm_image

def grayscale_to_rgb(image: torch.Tensor):
    '''
    transforms 1-channel grayscale image to rgb
    returns:
        - image with 3 channels
    '''
    if image.shape[-3] == 1:
        return image.repeat([3, 1, 1])
    else:
        return image

def truncate_thresholding(image: np.ndarray, threshold: float = 0.5, minus: bool = False):
    '''
    apply truncate thresholding to image
    - pixels brighter than the threshold will remain unchanged
    - pixels darker than the threshold will be zero
    inputs:
    - image: grayscale image pixel array
    - threshold: the threshold value
    - minus: whether to minus the threshold value of the pixels
    '''
    normal_threshold = lambda pixel: 0.0 if pixel < threshold else pixel
    minus_threshold = lambda pixel: max(pixel -  threshold, 0.0)
    # vectorize function, depending on whether 'minus' mode is used
    if minus:
        v_threshold = np.vectorize(minus_threshold)
    else:
        v_threshold = np.vectorize(normal_threshold)
    
    new_image = v_threshold(image)
    return new_image

def proportional_thresholding(image: np.ndarray, keep_proportion: float = 0.6):
    '''
    truncate thresholding that keeps the brightest portions of the image
    '''
    pixel_values = image.flatten()
    sample_size = int(len(pixel_values) * 0.1)
    threshold_pos = int(keep_proportion * sample_size)
    sample_pixels = np.random.choice(pixel_values, sample_size)
    sorted_values = np.sort(sample_pixels, kind='quicksort') # in ascending order
    threshold_value = sorted_values[-threshold_pos]
    truncate_threshold_img = truncate_thresholding(image, threshold_value)
    return truncate_threshold_img

def random_truncate_thresholding(
        image: np.ndarray,
        min_threshold: float = 0.3,
        max_threshold: float = 0.5
    ):
    '''
    truncate thresholding with a random threshold value
    inputs:
    - image: gray image pixel array
    - min_threshold: minimum value for random threshold
    - max_threshold: maximum value for random threshold
    '''
    threshold = min_threshold + np.random.random() * (max_threshold - min_threshold)
    new_image = truncate_thresholding(image, threshold)
    return new_image

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

# image transform objects
def default_transforms(
        to_rgb: bool = True, # whether to convert grayscale to 3-channel rgb
    ):
    '''
    get default transformation methods
    '''
    lst_default_transforms = [
        transforms.Lambda(normalize_image),
        transforms.ToTensor(),
    ]
    if to_rgb:
        lst_default_transforms.append(grayscale_to_rgb)
    default_image_transforms = transforms.Compose(lst_default_transforms)
    return default_image_transforms

# random_flip = transforms.RandomHorizontalFlip(p=0.5)
# padded_crop = transforms.RandomCrop(size=68, padding=4, probability=(0.8, 0.8, 0.1, 0.15))
# center_crop = transforms.CenterCrop(size=self.img_size)
# random_rotation = transforms.RandomRotation(degrees=7)
# random_resized_crop = transforms.RandomResizedCrop(
#         size=self.img_size, scale=(0.9, 1.0), ratio=(0.8, 0.9))
def image_augmentation(
        resize_size: tuple = (256, 256),
        crop_size: tuple = (224, 224)
        ):
    '''
    inputs:
    - resize_size: size after initial resizing
    - crop_size: size after center cropping
    '''
    image_augmentation_transforms = transforms.Compose([           
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=7),
        transforms.RandomResizedCrop(
            size=self.img_size, scale=(1.0, 1.1), ratio=(0.9, 1.0))
    ])

if __name__ == '__main__':
    print('debug...')
