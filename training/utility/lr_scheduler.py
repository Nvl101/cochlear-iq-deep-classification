'''
learning rate scheduler
'''
from abc import abstractclassmethod
from typing import Dict


class LearningRateScheduler:
    @abstractclassmethod
    def get_learning_rate(*args, **kwargs):
        pass
