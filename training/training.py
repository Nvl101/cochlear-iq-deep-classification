'''
abstract network training system, consisting of
- a neural network
- corresponding loss function
- corresponding optimizer
- a training dataloader
- a validation dataloader

functionality requirements:
- calculate validation loss and validation accuracy
- integrate displaying progress and file-controlled stop
- save neural network into file, and load neural network back
- output saliency map
- support parameter tuning
- support early stopping feature
'''
from abc import abstractclassmethod

class AbstractTraining:
    model: object
    criterion: object
    optimizer: object
    @abstractclassmethod
    def train(training_dataset, *args, **kwargs):
        pass
    @abstractclassmethod
    def validate(validation_dataset, *args, **kwargs):
        # yield accuracy and loss
        pass
    @abstractclassmethod
    def test(test_dataset, *args, **kwargs):
        pass
