'''
general classifier module

includes attributes: optimizers, loss calculations
methods: training, evaluation and prediction
'''
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from training.evaluation import LossEvaluator, AccuracyEvaluator
from training.utility.progress_bar import ProgressBar

class ClassifierModule:
    '''
    the general classifier module
    '''
    # attributes, to be specified in subclasses in the heading
    # core attributes for learning
    model: nn.Module
    criterion: nn.modules.loss
    optimizer: optim.Optimizer
    # auxiliary attributes for training process, None to skip
    progress_bar: ProgressBar = None
    scheduler: object = None
    early_stopper: object = None
    # parameters
    learning_rate: float
    current_epoch: int = 1
    verbose: bool = True
    def __init__(
            self,
            training_dataloader: DataLoader,
            validation_dataloader: DataLoader,
            test_dataloader: DataLoader,
            learning_rate: float = 0.0005,
            *args, **kwargs
        ):
        # learning 
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.learning_rate = learning_rate
        # loss and accuracyevaluators
        self.training_loss_evaluator = LossEvaluator(self.criterion)
        self.validation_loss_evaluator = LossEvaluator(self.criterion)
        self.validation_accuracy_evaluator = AccuracyEvaluator()
        self.test_accuracy_evaluator = AccuracyEvaluator()
        # auxiliary objects
        if self.verbose:
            self.progress_bar = ProgressBar(len(training_dataloader) + len(validation_dataloader))

    def print(self, *args, **kwargs):
        '''
        print message if verbose
        '''
        if self.verbose:
            print(*args, **kwargs)

    def train(self, epoches: int = 10):
        '''
        train model with number of epoches
        '''
        for _ in epoches:
            ##### INSERTED START
            print(f'epoch # {self.current_epoch}')
            epoch_start_time = time.time()
            # training network
            for images, labels in self.training_dataloader:
                labels_pred = self.model(images)
                # train loss
                loss_training = self.criterion(labels_pred, labels)
                self.training_loss_evaluator.append_loss(loss_training)
                # backpropagation
                self.optimizer.zero_grad()
                loss_training.backward()
                self.optimizer.step()
                self.progress_bar.step() if self.progress_bar else None
            # validating network
            for images, labels in self.validation_dataloader:
                images = images.to(torch.float32)
                labels = labels.to(torch.float32)
                labels_pred = self.model(images)
                loss_validation = self.criterion(labels_pred, labels)
                self.validation_loss_evaluator.append_loss(loss_validation)
                self.validation_accuracy_evaluator.append(labels_pred, labels)
                self.progress_bar.step() if self.progress_bar else None
            epoch_end_time = time.time()
            self.print(
                f'train loss: {self.training_loss_evaluator}  \
                validation loss: {self.validation_loss_evaluator}  \
                validation accuracy: {self.validation_accuracy_evaluator}')
            # updating the lr schedulers and early stoppers
            self.scheduler.step(self.validation_loss_evaluator.value()) if self.scheduler else None
            print(
            f'duration: {round(epoch_end_time - epoch_start_time, 2)} s  \
            learning rate: {round(self.scheduler.get_last_lr()[0], 9)}')
            self.early_stopper.step(self.validation_loss_evaluator.value()) if self.early_stopper else None
            self.validation_accuracy_evaluator.reset()
            ##### INSERTED END
            self.current_epoch += 1
    def predict(self, prediction_dataloader: DataLoader):
        '''
        make prediction on images in prediction_dataloader
        '''
        labels_pred = []
        for image in prediction_dataloader:
            label_pred = self.model(image)
            labels_pred.append(label_pred.item())
        return labels_pred