'''
classifier for 

consists of core methods: training, testing and prediction
and attributes:
- a neural network
- definition of loss function
- definition of optimizer and propagation methods
- fixed training and validation dataloaders


functionality requirements:
- calculate validation loss and validation accuracy
- integrate displaying progress and file-controlled stop
- save neural network into file, and load neural network back
- output saliency map
- support parameter tuning
- support early stopping feature
'''
from abc import abstractmethod
from typing import Any, Iterable, Dict, Type
from dataio.data_module import DataModule
from networks.classifier_module import ClassifierModule
from torch.utils.data import Dataset
import torch.nn as nn


class AbstractClassifier:
    '''
    abstract class for training model
    '''
    model: Module
    criterion: object
    optimizer: object
    # dataset attributes
    training_dataloader: Dataset
    validation_dataloader: Dataset
    test_set: Dataset

    @abstractmethod
    def train(self, epoches: int = 10, *args, **kwargs):
        pass
    @abstractmethod
    def predict(self, dicoms: Iterable[str], *args, **kwargs) -> Iterable:
        '''
        output prediction labels for dicom files
        '''
        pass
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        '''
        give a dictionary of evaluation values
        '''
        pass


class Classifier:
    '''
    classifier with basic functionality defined
    subclasses will inherit and define their own criterions and optimizers
    '''
    # neural network properties
    model: nn.Module
    criterion: object
    optimizer: object
    data: DataModule
    # datasets
    dicoms: Iterable[str]
    labels: Iterable
    # properties
    current_epoch: int = 1
    def __init__(
            self,
            data_module: DataModule,
            classifier_module: ClassifierModule
            # dicoms, labels,
            # data_module_type: DataModule = None
            # training_dicoms, training_labels, # split into training and validation
            # testing_dicoms, testing_labels,
            ):
        '''
        arguments:

        operations:
        1. create classifier model using the type parsed
        2. create data module using the type, with all dicoms and label types
        '''
        # self.model = self._get_model()
        # self.data = data_module_type(dicoms, labels)
        pass

    def train(self, epoches: int = 10, *args, **kwargs):
        for epoch in range(epoches):
            self.classifier_module.train()

    def predict(self, dicoms: Iterable[str], *args, **kwargs) -> Iterable:
        '''
        output prediction labels for dicom files
        '''
        prediction_dataloader = self.data_module.prediction_dataloader(dicoms)
        labels = self.classifier_module.predict(prediction_dataloader)
        return labels
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        '''
        give a dictionary of evaluation values
        '''
        raise NotImplementedError()
    @abstractmethod
    def _get_model(self, *args, **kwargs):
        raise NotImplementedError()
    def predict(self, dicoms: Iterable[str], *args, **kwargs) -> Iterable:
        '''
        output prediction labels for dicom files
        '''
        pred_dataset = self._get_dataloader(dicoms)
        labels_pred = self.model(pred_dataset)
        labels_pred_dn = self._denormalize_label(labels_pred)
        return labels_pred_dn

#TODO: move into a new script
from torch.optim import Adam
from networks.resnet_classifier import resnet18_classifier


class Resnet18Classifier(Classifier):
    '''
    classifier for resnet18 model
    '''
    def __init__(self, *args, **kwargs):
        self.super().__init__(*args, **kwargs)
        self.model = resnet18_classifier
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate) # TODO: 
        self.data = DataModule() # TODO: initiate the DataModule
    def train(self, n_epoches=10):
        # train using the training and validation data loaders
        for epoch in n_epoches:
            self.model.train()

    def predict(self, dicoms):
        pred_dataloader = self.create_dataloader(dicoms, type='pred')
        labels_pred = []
        for image in pred_dataloader:
            label_pred = self.model(image)
            labels_pred.append(label_pred)
        return labels_pred


    