'''
augmentation transforms
for cochlear image quality dataset

data augmentation strategies:
random crop, rotation, brightness change, 
'''
from torchvision import transforms

# TODO: define image augmentation algorithms
# NOTE: may cause size change, need to place resize in the end in dataset
image_augmentation = transforms.Compose(
    ...
)

