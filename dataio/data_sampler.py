'''
data sampler for training data loader
'''
from typing import Any, Dict, Iterable
from numpy.random import shuffle
from torch.utils.data.sampler import Sampler


def group_labels(labels: Iterable):
    '''
    group labels into a dictionary
    '''
    label_group = dict()
    for i, label in enumerate(labels):
        if label not in label_group.keys():
            label_group[label] = []
        label_group[label].append(i)
    return label_group

class RandomOverSampler(Sampler):
    '''
    Randomly sample the training dataset.
    This sampler can handle imbalanced dataset with oversampling.

    use in combination with dataset augmentation transforms
    '''
    label_group: Dict[Any, int]
    indices: Iterable[int]
    def __init__(self, labels: Iterable[Any], max_duplicates: int = 2):
        '''
        Oversample from imbalanced data labels.

        This is done by multiplying smaller datasets N times.
        
        where `N = min(largest_group_size // current_group_size, max_duplicates + 1)`

        arguments:
            - labels: labels used in training dataset, inserted in correct order
            - max_duplicates: maximum times to duplicate data in smaller sets,
            excluding the original.
        '''
        label_group = group_labels(labels)
        # dictionary shows group with {key: length}
        group_length = {key: len(lst) for key, lst in label_group.items()}
        largest_group_size = max([length for _, length in group_length.items()])
        # extend each label group N times, and extend self.indices
        self.indices = []
        for key, lbl in label_group.items():
            current_group_size = group_length[key]
            n_duplicates = min(largest_group_size // current_group_size, max_duplicates + 1)
            self.indices.extend(lbl * n_duplicates)
        # finally, shuffle the indices randomly
        shuffle(self.indices)
    def __len__(self):
        return len(self.indices)
    def __iter__(self):
        for index in self.indices:
            yield index
